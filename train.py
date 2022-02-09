import argparse
import os

import torch
import yaml
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp

from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, to_device, log, synth_one_sample
from model import PortaSpeechLoss
from dataset import Dataset

from evaluate import evaluate

torch.backends.cudnn.benchmark = True


def train(rank, args, configs, batch_size, num_gpus):
    preprocess_config, model_config, train_config = configs
    if num_gpus > 1:
        init_process_group(
            backend=train_config["dist_config"]['dist_backend'],
            init_method=train_config["dist_config"]['dist_url'],
            world_size=train_config["dist_config"]['world_size'] * num_gpus,
            rank=rank,
        )
    device = torch.device('cuda:{:d}'.format(rank))

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, model_config, train_config, sort=True, drop_last=True
    )
    data_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)
    scaler = amp.GradScaler(enabled=args.use_amp)
    Loss = PortaSpeechLoss(
        preprocess_config, model_config, train_config).to(device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    if rank == 0:
        print("Number of PortaSpeech Parameters: {}\n".format(get_param_num(model)))
        # Init logger
        for p in train_config["path"].values():
            os.makedirs(p, exist_ok=True)
        train_log_path = os.path.join(
            train_config["path"]["log_path"], "train")
        val_log_path = os.path.join(train_config["path"]["log_path"], "val")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)

        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        outer_bar.n = args.restore_step
        outer_bar.update()

    train = True
    while train:
        if rank == 0:
            inner_bar = tqdm(total=len(loader),
                             desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batchs in loader:
            if train == False:
                break
            for batch in batchs:
                batch = to_device(batch, device)

                with amp.autocast(args.use_amp):
                    # Forward
                    output = model(*(batch[2:]))

                    # Cal Loss
                    losses = Loss(batch, output, step)
                    total_loss = losses[0]
                    total_loss = total_loss / grad_acc_step

                # Backward
                scaler.scale(total_loss).backward()

                # Clipping gradients to avoid gradient explosion
                if step % grad_acc_step == 0:
                    scaler.unscale_(optimizer._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip_thresh)

                # Update weights
                optimizer.step_and_update_lr(scaler)
                scaler.update()
                optimizer.zero_grad()

                if rank == 0:
                    if step % log_step == 0:
                        losses = [l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, KL Loss: {:.4f}, PN Loss: {:.4f}, Duration Loss: {:.4f}, Helper Loss: {:.4f}".format(
                            *losses
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(train_logger, step, losses=losses)

                    if step % synth_step == 0:
                        fig, attn_fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            model,
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                            num_gpus,
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        log(
                            train_logger,
                            fig=attn_fig,
                            tag="Training_attn/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(
                                step, tag),
                        )
                        log(
                            train_logger,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(
                                step, tag),
                        )

                    if step % val_step == 0:
                        model.eval()
                        message = evaluate(
                            device, model, step, configs, val_logger, vocoder, len(losses), num_gpus)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()

                    if step % save_step == 0:
                        torch.save(
                            {
                                "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                                "optimizer": optimizer._optimizer.state_dict(),
                            },
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    train = False
                    break
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(
        args.dataset)
    configs = (preprocess_config, model_config, train_config)

    # Set Device
    torch.manual_seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"])
    num_gpus = torch.cuda.device_count()
    batch_size = int(train_config["optimizer"]["batch_size"] / num_gpus)
    helper_type = train_config["aligner"]["helper_type"]

    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(' ---> Automatic Mixed Precision:', args.use_amp)
    print(' ---> Number of used GPU:', num_gpus)
    print(' ---> Batch size per GPU:', batch_size)
    print(' ---> Batch size in total:', batch_size * num_gpus)
    print(' ---> Type of alignment helper:', helper_type)
    print("=================================================================================================")
    print("Prepare training ...")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(
            args, configs, batch_size, num_gpus))
    else:
        train(0, args, configs, batch_size, num_gpus)

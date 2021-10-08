import math
import torch
import torch.nn as nn


class PortaSpeechLoss(nn.Module):
    """ PortaSpeech Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(PortaSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def kl_loss(self, z_p, logs_q, mask):
        """
        z_p, logs_q: [batch_size, dim, max_time]
        mask -- [batch_size, 1, max_time]
        """
        m_p, logs_p = torch.zeros_like(z_p), torch.zeros_like(z_p)
        z_p = z_p.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        logs_q = logs_q.float()
        mask = mask.float()

        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
        kl = torch.sum(kl * mask)
        l = kl / torch.sum(mask)
        return l

    def mle_loss(self, z, logdet, mask):
        """
        z, logdet: [batch_size, dim, max_time]
        mask -- [batch_size, 1, max_time]
        """
        logs = torch.zeros_like(z * mask)
        l = torch.sum(logs) + 0.5 * \
            torch.sum(torch.exp(-2 * logs) * (z**2))
        l = l - torch.sum(logdet)
        l = l / \
            torch.sum(torch.ones_like(z * mask))
        l = l + 0.5 * math.log(2 * math.pi)
        return l

    def forward(self, inputs, predictions):
        (
            mel_targets,
            *_,
        ) = inputs[10:]
        (
            mel_predictions,
            postnet_outputs,
            log_duration_predictions,
            duration_roundeds,
            src_masks,
            mel_masks,
            _,
            _,
            _,
            dist_info,
            src_w_masks,
            _,
        ) = predictions
        log_duration_targets = torch.log(duration_roundeds.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        mel_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(
            src_w_masks)
        log_duration_targets = log_duration_targets.masked_select(src_w_masks)

        mel_predictions = mel_predictions.masked_select(
            mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)

        duration_loss = self.mse_loss(
            log_duration_predictions, log_duration_targets)

        kl_loss = self.kl_loss(*dist_info)

        z, logdet = postnet_outputs
        postnet_loss = self.mle_loss(z, logdet, mel_masks.unsqueeze(1))

        total_loss = (
            mel_loss + kl_loss + postnet_loss + duration_loss
        )

        return (
            total_loss,
            mel_loss,  # L_VG
            kl_loss,  # L_KL
            postnet_loss,  # L_PN
            duration_loss,  # L_dur
        )

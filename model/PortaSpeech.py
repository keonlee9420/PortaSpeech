# import os
# import json

import torch
import torch.nn as nn

from utils.tools import get_mask_from_lengths
from .linguistic_encoder import LinguisticEncoder
from .variational_generator import VariationalGenerator
from .postnet import FlowPostNet


class PortaSpeech(nn.Module):
    """ PortaSpeech """

    def __init__(self, preprocess_config, model_config):
        super(PortaSpeech, self).__init__()
        self.model_config = model_config

        self.linguistic_encoder = LinguisticEncoder(model_config, abs_mha=True)
        self.variational_generator = VariationalGenerator(
            preprocess_config, model_config)
        self.postnet = FlowPostNet(preprocess_config, model_config)

        # self.speaker_emb = None
        # if model_config["multi_speaker"]:
        #     self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
        #     if self.embedder_type == "none":
        #         with open(
        #             os.path.join(
        #                 preprocess_config["path"]["preprocessed_path"], "speakers.json"
        #             ),
        #             "r",
        #         ) as f:
        #             n_speaker = len(json.load(f))
        #         self.speaker_emb = nn.Embedding(
        #             n_speaker,
        #             model_config["transformer"]["encoder_hidden"],
        #         )
        #     else:
        #         self.speaker_emb = nn.Linear(
        #             model_config["external_speaker_dim"],
        #             model_config["transformer"]["encoder_hidden"],
        #         )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        word_boundaries,
        src_w_lens,
        max_src_w_len,
        spker_embeds=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        d_targets=None,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        src_w_masks = get_mask_from_lengths(src_w_lens, max_src_w_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        (
            output,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            alignments,
        ) = self.linguistic_encoder(
            texts,
            src_lens,
            word_boundaries,
            src_masks,
            src_w_lens,
            src_w_masks,
            mel_masks,
            max_mel_len,
            d_targets,
            d_control,
        )

        # if self.speaker_emb is not None:
        #     if self.embedder_type == "none":
        #         output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        #             -1, max_src_len, -1
        #         )
        #     else:
        #         assert spker_embeds is not None, "Speaker embedding should not be None"
        #         output = output + self.speaker_emb(spker_embeds).unsqueeze(1).expand(
        #             -1, max_src_len, -1
        #         )

        residual = output
        if mels is not None:
            output, out_residual, dist_info = self.variational_generator(
                mels, mel_lens, mel_masks, output)
            postnet_output = self.postnet(
                mels.transpose(1, 2),
                ~mel_masks.unsqueeze(1),
                g=(out_residual + residual).transpose(1, 2),
            )
        else:
            _, out_residual, dist_info = self.variational_generator.inference(
                mel_lens, mel_masks, output)
            output = self.postnet.inference(
                ~mel_masks.unsqueeze(1),
                g=(out_residual + residual).transpose(1, 2),
            )
            postnet_output = None

        return (
            output,
            postnet_output,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            alignments,
            dist_info,
            src_w_masks,
            residual,
        )

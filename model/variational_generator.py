import math

import torch
import torch.nn as nn
import numpy as np

from utils.tools import get_mask_from_lengths, pad, reparameterize

from .blocks import (
    Flip,
    LinearNorm,
    ConvBlock,
    ConvTransposeBlock,
    NonCausalWaveNet,
)


class VariationalGenerator(nn.Module):
    """ Variational Generator """

    def __init__(self, preprocess_config, model_config):
        super(VariationalGenerator, self).__init__()

        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_model = model_config["transformer"]["encoder_hidden"]
        encoder_layer = model_config["variational_generator"]["encoder_layer"]
        decoder_layer = model_config["variational_generator"]["decoder_layer"]
        conv_kernel_size = model_config["variational_generator"]["conv_kernel_size"]
        conv_stride_size = model_config["variational_generator"]["conv_stride_size"]
        encoder_decoder_hidden = model_config["variational_generator"]["encoder_decoder_hidden"]
        # encoder_decoder_dropout = model_config["variational_generator"]["encoder_decoder_dropout"]
        latent_hidden = model_config["variational_generator"]["latent_hidden"]
        flow_layer = model_config["variational_generator"]["vp_flow_layer"]
        flow_hidden = model_config["variational_generator"]["vp_flow_hidden"]
        flow_kernel = model_config["variational_generator"]["vp_flow_kernel"]

        dilation = 1
        self.padding_size = int(
            dilation * (conv_kernel_size - conv_stride_size) / 2)
        self.latent_hidden = latent_hidden
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride_size = conv_stride_size

        self.n_mel_channels = n_mel_channels

        self.enc_conv = ConvBlock(
            n_mel_channels,
            encoder_decoder_hidden,
            conv_kernel_size,
            stride=conv_stride_size,
            padding=self.padding_size,
            dilation=dilation,
            w_init_gain="linear",
            # dropout=encoder_decoder_dropout,
            activation=None,
            layer_norm=False,
        )
        self.cond_layer_e = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                d_model,
                2*encoder_decoder_hidden*encoder_layer,  # d_model
                kernel_size=conv_kernel_size,
                stride=conv_stride_size,
                padding=self.padding_size,
                dilation=dilation), name='weight')
        # self.cond_layer_e_prj = LinearNorm(
        #     d_model, 2*encoder_decoder_hidden*encoder_layer)
        self.enc_wn = NonCausalWaveNet(
            encoder_decoder_hidden,
            conv_kernel_size,
            1,
            encoder_layer,
            d_model,
        )
        self.latent_enc_prj = LinearNorm(
            encoder_decoder_hidden, latent_hidden * 2)
        self.cond_layer_f = nn.Conv1d(
            d_model,
            encoder_decoder_hidden,
            kernel_size=conv_kernel_size,
            stride=conv_stride_size,
            padding=self.padding_size,
            dilation=dilation,
        )
        self.flow = VPFlow(
            latent_hidden,
            flow_hidden,
            flow_kernel,
            1,
            flow_layer,
            gin_channels=d_model
        )
        self.latent_dec_prj = LinearNorm(latent_hidden, encoder_decoder_hidden)
        self.cond_layer_d = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                d_model,
                2*encoder_decoder_hidden*decoder_layer,  # d_model
                kernel_size=conv_kernel_size,
                stride=conv_stride_size,
                padding=self.padding_size,
                dilation=dilation), name='weight')
        # self.cond_layer_d_prj = LinearNorm(
        #     d_model, 2*encoder_decoder_hidden*decoder_layer)
        self.dec_wn = NonCausalWaveNet(
            encoder_decoder_hidden,
            conv_kernel_size,
            1,
            decoder_layer,
            d_model,
        )
        self.dec_conv = ConvTransposeBlock(
            encoder_decoder_hidden,
            n_mel_channels,
            conv_kernel_size,
            stride=conv_stride_size,
            padding=self.padding_size,
            output_padding=self.conv_stride_size - 1,  # for preserving sequence length
            dilation=dilation,
            w_init_gain="linear",
            # dropout=encoder_decoder_dropout,
            activation=None,
            layer_norm=False,
        )
        self.residual_layer = LinearNorm(n_mel_channels, d_model)

    def pad_input(self, x):
        """
        x -- [batch_size, max_time, dim]
        """
        if self.conv_stride_size == 1:
            return x
        pad_size = x.shape[1] + 2 * self.padding_size - \
            1 * (self.conv_kernel_size - 1)
        pad_size = pad_size % self.conv_stride_size
        if pad_size > 0:
            pad = torch.zeros(x.shape[0], pad_size, x.shape[2]).to(
                device=x.device, dtype=x.dtype)
            x = torch.cat((x, pad), dim=1)
        return x

    def get_conv_mask(self, mel_len, tgt_max_len, mel_mask):
        """
        mel_len -- [batch_size,]
        mask_conv -- [batch_size, max_time]
        """
        if max(mel_len).item() == tgt_max_len:
            return mel_mask
        mel_len_conv = []
        for ml in mel_len:
            pad_size = ml + 2 * self.padding_size - \
                1 * (self.conv_kernel_size - 1)
            pad_size = pad_size % self.conv_stride_size
            ml_conv = math.floor(((ml + pad_size) + 2 * self.padding_size - 1 * (
                self.conv_kernel_size - 1)) / self.conv_stride_size + 1)
            # no way to remove this?
            mel_len_conv.append(min(ml_conv, tgt_max_len))
        mask_conv = get_mask_from_lengths(torch.tensor(
            mel_len_conv, dtype=mel_len.dtype, device=mel_len.device))
        return mask_conv

    def trim_output(self, x, max_len):
        return x[:, :max_len]

    def forward(self, mel, mel_len, mel_mask, h_text):
        """
        mel -- [batch_size, max_time, n_mels]
        mel_len -- [batch_size,]
        mel_mask -- [batch_size, max_time]
        h_text -- [batch_size, max_time, dim]
        """
        h_text = self.pad_input(h_text)
        mel = self.pad_input(mel)

        # Prepare Conditioner
        # h_text_f = self.cond_layer_f(h_text.transpose(1, 2))  # [B, H, L']
        # h_text_e = self.cond_layer_e_prj(self.cond_layer_e(
        #     h_text.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)  # [B, H, L']
        # h_text_d = self.cond_layer_d_prj(self.cond_layer_d(
        #     h_text.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)  # [B, H, L']
        h_text = h_text.transpose(1, 2)
        h_text_f = self.cond_layer_f(h_text)  # [B, H, L']
        h_text_e = self.cond_layer_e(h_text)  # [B, H, L']
        h_text_d = self.cond_layer_d(h_text)  # [B, H, L']
        mel_mask_conv = self.get_conv_mask(
            mel_len, h_text_f.shape[2], mel_mask).unsqueeze(-1)

        # Encoding
        x = self.enc_conv(mel)
        x = x.contiguous().transpose(1, 2)
        x = self.enc_wn(x, g=h_text_e) * mel_mask_conv.transpose(1, 2)
        x = x.contiguous().transpose(1, 2)
        x = self.latent_enc_prj(x)

        # # Reparameterization
        m_q, logs_q = torch.split(x, self.latent_hidden, dim=-1)
        m_q, logs_q = m_q * mel_mask_conv, logs_q * mel_mask_conv
        z_q = reparameterize(m_q, logs_q) * mel_mask_conv

        # Prior VP FLow
        z_p = self.flow(z_q.transpose(1, 2), x_mask=mel_mask_conv.transpose(
            1, 2), g=h_text_f, reverse=False)

        # Decoding
        x = self.latent_dec_prj(z_q)
        x = x.contiguous().transpose(1, 2)
        x = self.dec_wn(x, g=h_text_d) * mel_mask_conv.transpose(1, 2)
        x = x.contiguous().transpose(1, 2)
        mel_res = self.dec_conv(x)
        mel_res = self.trim_output(
            mel_res, mel_mask.shape[1]) * mel_mask.unsqueeze(-1)
        residual = self.residual_layer(mel_res) * mel_mask.unsqueeze(-1)

        return mel_res, residual, (z_p, logs_q.transpose(1, 2), mel_mask_conv.transpose(1, 2))

    def inference(self, mel_len, mel_mask, h_text):
        """
        mel_len -- [batch_size,]
        mel_mask -- [batch_size, max_time]
        h_text -- [batch_size, max_time, dim]
        """
        batch_size, device, dtype = h_text.shape[0], h_text.device, h_text.dtype
        h_text = self.pad_input(h_text)

        # Prepare Conditioner
        # h_text_f = self.cond_layer_f(h_text.transpose(1, 2))  # [B, H, L']
        # h_text_d = self.cond_layer_d_prj(self.cond_layer_d(
        #     h_text.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)  # [B, H, L']
        h_text = h_text.transpose(1, 2)
        h_text_f = self.cond_layer_f(h_text)  # [B, H, L']
        h_text_d = self.cond_layer_d(h_text)  # [B, H, L']
        mel_mask_conv = self.get_conv_mask(
            mel_len, h_text_f.shape[2], mel_mask).unsqueeze(-1)

        # Sample from Prior
        z_n = torch.randn(h_text_f.shape[0], self.latent_hidden,
                          h_text_f.shape[2]).to(device=h_text_f.device, dtype=h_text.dtype)
        z_q = self.flow(z_n, x_mask=mel_mask_conv.transpose(
            1, 2), g=h_text_f, reverse=True)

        # Decoding
        x = self.latent_dec_prj(z_q.transpose(1, 2))
        x = x.contiguous().transpose(1, 2)
        x = self.dec_wn(x, g=h_text_d) * mel_mask_conv.transpose(1, 2)
        x = x.contiguous().transpose(1, 2)
        mel_res = self.dec_conv(x)
        mel_res = self.trim_output(
            mel_res, mel_mask.shape[1]) * mel_mask.unsqueeze(-1)
        residual = self.residual_layer(mel_res) * mel_mask.unsqueeze(-1)

        return mel_res, residual, None


class VPFlow(nn.Module):
    """ Volume-Preserving Normalizing Flow """

    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size,
                              dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, x_mask=None, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 p_dropout=0,
                 gin_channels=0,
                 mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.cond_layer = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                gin_channels, 2*hidden_channels*n_layers, 1), name='weight')
        self.enc = NonCausalWaveNet(hidden_channels, kernel_size, dilation_rate,
                                    n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(
            hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask=None, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        if x_mask is None:
            x_mask = 1
        h = self.pre(x0) * x_mask
        if g is not None:
            g = self.cond_layer(g)
        h = self.enc(h, x_mask=x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels]*2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

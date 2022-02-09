import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PortaSpeechLoss(nn.Module):
    """ PortaSpeech Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(PortaSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.helper_type = train_config["aligner"]["helper_type"]
        if self.helper_type == "dga":
            self.guided_attn_loss = GuidedAttentionLoss(
                sigma=train_config["aligner"]["guided_sigma"],
                alpha=train_config["aligner"]["guided_lambda"],
            )
            self.guided_attn_weight = train_config["aligner"]["guided_weight"]
        elif self.helper_type == "ctc":
            self.sum_loss = ForwardSumLoss()
            self.ctc_step = train_config["step"]["ctc_step"]
            self.ctc_weight_start = train_config["aligner"]["ctc_weight_start"]
            self.ctc_weight_end = train_config["aligner"]["ctc_weight_end"]

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

    def forward(self, inputs, predictions, step):
        (
            mel_targets,
            *_,
        ) = inputs[11:]
        (
            mel_predictions,
            postnet_outputs,
            log_duration_predictions,
            duration_roundeds,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            alignments,
            dist_info,
            src_w_masks,
            _,
            alignment_logprobs,
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

        helper_loss = attn_loss = ctc_loss = torch.zeros(1).to(mel_targets.device)
        if self.helper_type == "dga":
            for alignment in alignments[1]: # DGA should be applied on attention without mapping mask
                attn_loss += self.guided_attn_loss(alignment, src_lens, mel_lens)
            # attn_loss = self.guided_attn_loss(alignments[1][0], src_lens, mel_lens)
            helper_loss = self.guided_attn_weight * attn_loss
        elif self.helper_type == "ctc":
            for alignment_logprob in alignment_logprobs:
                ctc_loss += self.sum_loss(alignment_logprob, src_lens, mel_lens)
            ctc_loss = ctc_loss.mean()
            helper_loss = (self.ctc_weight_start if step <= self.ctc_step else self.ctc_weight_end) * ctc_loss

        total_loss = (
            mel_loss + kl_loss + postnet_loss + duration_loss + helper_loss
        )

        return (
            total_loss,
            mel_loss,  # L_VG
            kl_loss,  # L_KL
            postnet_loss,  # L_PN
            duration_loss,  # L_dur
            helper_loss,
        )


class GuidedAttentionLoss(nn.Module):
    """Guided attention loss function module.
    See https://github.com/espnet/espnet/blob/e962a3c609ad535cd7fb9649f9f9e9e0a2a27291/espnet/nets/pytorch_backend/e2e_tts_tacotron2.py#L25
    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    def _make_masks(self, ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        in_masks = self.make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = self.make_non_pad_mask(olens)  # (B, T_out)
        # (B, T_out, T_in)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)

    def make_non_pad_mask(self, lengths, xs=None, length_dim=-1):
        return ~self.make_pad_mask(lengths, xs, length_dim)

    def make_pad_mask(self, lengths, xs=None, length_dim=-1):
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)

        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            # ind = (:, None, ..., None, :, , None, ..., None)
            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            mask = mask[ind].expand_as(xs).to(xs.device)
        return mask


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss

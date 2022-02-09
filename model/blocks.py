import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


class Flip(nn.Module):
    """ Flip Layer """

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    """ 1D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, w_init_gain=None, dropout=None, activation=nn.ReLU, layer_norm=True):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                w_init_gain=w_init_gain,
                channel_last=True,
            ),
            activation() if activation is not None else nn.Identity(),
            nn.LayerNorm(out_channels) if layer_norm else nn.Identity(),
        )
        self.dropout = dropout if dropout is not None else None

    def forward(self, enc_input, mask=None):
        enc_output = self.conv_layer(enc_input)
        if self.dropout is not None:
            enc_output = F.dropout(enc_output, self.dropout, self.training)

        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain=None,
        channel_last=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if w_init_gain is not None:
            torch.nn.init.xavier_uniform_(
                self.conv.weight, gain=torch.nn.init.calculate_gain(
                    w_init_gain)
            )
        self.channel_last = channel_last

    def forward(self, x):
        if self.channel_last:
            x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        if self.channel_last:
            x = x.contiguous().transpose(1, 2)

        return x


class ConvTransposeBlock(nn.Module):
    """ 1D Transposed Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, output_padding=0, w_init_gain=None, dropout=None, activation=nn.ReLU, layer_norm=True):
        super(ConvTransposeBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvTransposeNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                dilation=dilation,
                w_init_gain=w_init_gain,
                channel_last=True
            ),
            activation() if activation is not None else nn.Identity(),
            nn.LayerNorm(out_channels) if layer_norm else nn.Identity(),
        )
        self.dropout = dropout if dropout is not None else None

    def forward(self, enc_input, mask=None):
        enc_output = self.conv_layer(enc_input)
        if self.dropout is not None:
            enc_output = F.dropout(enc_output, self.dropout, self.training)

        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvTransposeNorm(nn.Module):
    """ 1D Transposed Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        output_padding=None,
        dilation=1,
        bias=True,
        w_init_gain=None,
        channel_last=False,
    ):
        super(ConvTransposeNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias,
        )

        if w_init_gain is not None:
            torch.nn.init.xavier_uniform_(
                self.conv.weight, gain=torch.nn.init.calculate_gain(
                    w_init_gain)
            )
        self.channel_last = channel_last

    def forward(self, x):
        if self.channel_last:
            x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        if self.channel_last:
            x = x.contiguous().transpose(1, 2)

        return x


class NonCausalWaveNet(torch.nn.Module):
    """ Non-Causal WaveNet """

    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, n_sqz=1):
        super(NonCausalWaveNet, self).__init__()
        assert(kernel_size % 2 == 1)
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        # self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(
                hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g=None):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset+2*self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(
                x_in,
                g_l,
                n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts)
                if x_mask is not None:
                    x = x * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        if x_mask is not None:
            output = output * x_mask
        return output

    def remove_weight_norm(self):
        # if self.gin_channels != 0:
        #     torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class RelativeFFTBlock(nn.Module):
    """ FFT Block with Relative Multi-Head Attention """

    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=None, block_length=None):
        super(RelativeFFTBlock, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(RelativeSelfAttention(hidden_channels, hidden_channels, n_heads,
                                    window_size=window_size, p_dropout=p_dropout, block_length=block_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(
                hidden_channels, hidden_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class RelativeSelfAttention(nn.Module):
    """ Relative Multi-Head Attention """

    def __init__(self, channels, out_channels, n_heads, window_size=None, heads_share=True, p_dropout=0., block_length=None, proximal_bias=False, proximal_init=False):
        super(RelativeSelfAttention, self).__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(torch.randn(
                n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(
                n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels,
                           t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels,
                           t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(self.k_channels)
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(
                rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + \
                self._attention_bias_proximal(t_s).to(
                    device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                block_mask = torch.ones_like(
                    scores).triu(-self.block_length).tril(self.block_length)
                scores = scores * block_mask + -1e4*(1 - block_mask)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s)
            output = output + \
                self._matmul_with_relative_values(
                    relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(
            b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                                              slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape(
            [[0, 0], [0, 0], [0, length-1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view(
            [batch, heads, length+1, 2*length-1])[:, :, :length, length-1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, convert_pad_shape(
            [[0, 0], [0, 0], [0, 0], [0, length-1]]))
        x_flat = x.view([batch, heads, length**2 + length*(length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, convert_pad_shape(
            [[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2*length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """
        Bias for self-attention to encourage attention to close positions.
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean)**2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, p_dropout=0., activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv(x * x_mask)
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        return x * x_mask


class WordToPhonemeAttention(nn.Module):
    """ Word-to-Phoneme Multi-Head Attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.0):
        super(WordToPhonemeAttention, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = LinearNorm(d_model, n_head * d_k)
        self.w_ks = LinearNorm(d_model, n_head * d_k)
        self.w_vs = LinearNorm(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        # self.layer_norm = nn.LayerNorm(d_model)

        self.fc = LinearNorm(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_mask=None, query_mask=None, mapping_mask=None, indivisual_attn=False, attn_prior=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        if key_mask is not None:
            key_mask = key_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        if query_mask is not None:
            query_mask = query_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        if mapping_mask is not None:
            mapping_mask = mapping_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        if attn_prior is not None:
            attn_prior = attn_prior.repeat(n_head, 1, 1)
        output, attns, attn_logprob = self.attention(
            q, k, v, key_mask=key_mask, query_mask=query_mask, mapping_mask=mapping_mask, attn_prior=attn_prior)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (
            output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        )  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        # output = self.layer_norm(output)

        if indivisual_attn:
            attns = tuple([attn.view(n_head, sz_b, len_q, len_k) for attn in attns])
            attn_logprob = attn_logprob.view(n_head, sz_b, 1, len_q, len_k)

        return output, attns, attn_logprob


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)

    def forward(self, q, k, v, key_mask=None, query_mask=None, mapping_mask=None, attn_prior=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if key_mask is not None:
            attn = attn.masked_fill(key_mask == 0., -np.inf)
        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior.transpose(1, 2) + 1e-8)
        attn_logprob = attn.unsqueeze(1).clone()

        attn = self.softmax(attn)

        if query_mask is not None:
            attn = attn * query_mask
        attn_raw = attn.clone()
        if mapping_mask is not None:
            attn = attn * mapping_mask
        output = torch.bmm(attn, v)

        return output, (attn, attn_raw), attn_logprob

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import NonCausalWaveNet


def pad_input(x, x_mask=None, g=None, n_sqz=2):
    """
    x -- [batch_size, dim, max_time]
    x_mask -- [batch_size, 1, max_time]
    g -- [batch_size, dim, max_time]
    """
    pad_size = x.shape[2] % n_sqz
    if pad_size > 0:
        pad = torch.zeros(x_mask.shape[0], 1, pad_size).to(
            device=x_mask.device, dtype=x_mask.dtype)
        x = torch.cat((x, pad.repeat(1, x.shape[1], 1)), dim=2)
        if x_mask is not None:
            x_mask = torch.cat((x_mask, pad), dim=2)
        if g is not None:
            g = torch.cat((g, pad.repeat(1, g.shape[1], 1)), dim=2)
    return x, x_mask, g, pad_size


def squeeze(x, x_mask=None, g=None, n_sqz=2):
    x, x_mask, g, pad_size = pad_input(x, x_mask, g, n_sqz)
    b, c_x, t_x = x.size()
    t_x = (t_x // n_sqz) * n_sqz
    x = x[:, :, :t_x]
    x_sqz = x.view(b, c_x, t_x//n_sqz, n_sqz)
    x_sqz = x_sqz.permute(0, 3, 1, 2).contiguous().view(
        b, c_x*n_sqz, t_x//n_sqz)

    g_sqz = None
    if g is not None:
        _, c_g, t_x = g.size()
        g = g[:, :, :t_x]
        g_sqz = g.view(b, c_g, t_x//n_sqz, n_sqz)
        g_sqz = g_sqz.permute(0, 3, 1, 2).contiguous().view(
            b, c_g*n_sqz, t_x//n_sqz)

    if x_mask is not None:
        x_mask = x_mask[:, :, n_sqz-1::n_sqz]
    else:
        x_mask = torch.ones(
            b, 1, t_x//n_sqz).to(device=x.device, dtype=x.dtype)
    return x_sqz * x_mask, g_sqz * x_mask, x_mask, pad_size


def unsqueeze(x, x_mask=None, g=None, n_sqz=2, pad_size=0):
    b, c_x, t_x = x.size()

    x_unsqz = x.view(b, n_sqz, c_x//n_sqz, t_x)
    x_unsqz = x_unsqz.permute(
        0, 2, 3, 1).contiguous().view(b, c_x//n_sqz, t_x*n_sqz)

    g_unsqz = None
    if g is not None:
        _, c_g, t_x = g.size()

        g_unsqz = g.view(b, n_sqz, c_g//n_sqz, t_x)
        g_unsqz = g_unsqz.permute(
            0, 2, 3, 1).contiguous().view(b, c_g//n_sqz, t_x*n_sqz)

    if x_mask is not None:
        x_mask = x_mask.unsqueeze(-1).repeat(1, 1, 1,
                                             n_sqz).view(b, 1, t_x*n_sqz)
    else:
        x_mask = torch.ones(b, 1, t_x*n_sqz).to(device=x.device, dtype=x.dtype)

    if pad_size > 0:
        x_unsqz = x_unsqz[:, :, :-pad_size]
        if x_mask is not None:
            x_mask = x_mask[:, :, :-pad_size]
        if g is not None:
            g_unsqz = g_unsqz[:, :, :-pad_size]

    return x_unsqz * x_mask, g_unsqz * x_mask, x_mask


class FlowPostNet(nn.Module):
    """ Flow-based Post-Net with grouped parameter sharing """

    def __init__(self, preprocess_config, model_config):
        super(FlowPostNet, self).__init__()

        in_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        hidden_channels = model_config["postnet"]["wn_hidden"]
        kernel_size = model_config["postnet"]["wn_kernel_size"]
        dilation_rate = 1
        n_blocks = model_config["postnet"]["flow_step"]
        n_layers = model_config["postnet"]["wn_layer"]
        p_dropout = model_config["postnet"]["flow_dropout"]
        n_split = model_config["postnet"]["n_split"]
        n_sqz = model_config["postnet"]["n_sqz"]
        sigmoid_scale = model_config["postnet"]["sigmoid_scale"]
        gin_channels = model_config["transformer"]["encoder_hidden"]
        shared_group = model_config["postnet"]["shared_group"]

        self.in_channels = in_channels
        self.n_sqz = n_sqz
        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            create_wn = b % shared_group == 0
            if create_wn:
                shared_wn_idx = b
            self.flows.append(ActNorm(channels=in_channels * n_sqz))
            self.flows.append(InvConvNear(
                channels=in_channels * n_sqz, n_split=n_split))
            self.flows.append(
                CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                    n_sqz=n_sqz,
                    shared_wn=None if create_wn else self.flows[3*shared_wn_idx+2].wn))

    def forward(self, x, x_mask, g=None):
        flows = self.flows
        logdet_tot = 0

        if self.n_sqz > 1:
            x, g, x_mask, pad_size = squeeze(x, x_mask, g, self.n_sqz)
        for f in flows:
            x, logdet = f(x, x_mask, g=g, reverse=False)
            logdet_tot += logdet
        if self.n_sqz > 1:
            x, g, x_mask = unsqueeze(x, x_mask, g, self.n_sqz, pad_size)

        return x, logdet_tot

    def inference(self, x_mask, g=None, temperature=0.8):
        flows = reversed(self.flows)
        logdet_tot = None

        # sampling from gaussian noise
        x = torch.normal(0.0, temperature, [x_mask.shape[0], self.in_channels, x_mask.shape[2]]).to(
            x_mask.device)

        if self.n_sqz > 1:
            x, g, x_mask, pad_size = squeeze(x, x_mask, g, self.n_sqz)
        for f in flows:
            x, logdet = f(x, x_mask, g=g, reverse=True)
        if self.n_sqz > 1:
            x, g, x_mask = unsqueeze(x, x_mask, g, self.n_sqz, pad_size)

        return x

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()


class CouplingBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, sigmoid_scale=False, n_sqz=1, shared_wn=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        self.sigmoid_scale = sigmoid_scale

        start = torch.nn.Conv1d(in_channels//2, hidden_channels, 1)
        start = torch.nn.utils.weight_norm(start)
        self.start = start
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  It helps to stabilze training.
        end = torch.nn.Conv1d(hidden_channels, in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        self.cond_layer = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                n_sqz * gin_channels, 2*hidden_channels*n_layers, 1), name='weight')

        # Grouped Parameter Sharing
        self.wn = shared_wn if shared_wn is not None else NonCausalWaveNet(hidden_channels, kernel_size,
                                                                           dilation_rate, n_layers, gin_channels, p_dropout, n_sqz)

    def forward(self, x, x_mask=None, reverse=False, g=None, **kwargs):
        b, c, t = x.size()
        if x_mask is None:
            x_mask = 1
        x_0, x_1 = x[:, :self.in_channels//2], x[:, self.in_channels//2:]

        x = self.start(x_0) * x_mask
        if g is not None:
            g = self.cond_layer(g)
        x = self.wn(x, x_mask=x_mask, g=g)
        out = self.end(x)

        z_0 = x_0
        m = out[:, :self.in_channels//2, :]
        logs = out[:, self.in_channels//2:, :]
        if self.sigmoid_scale:
            logs = torch.log(1e-6 + torch.sigmoid(logs + 2))

        if reverse:
            z_1 = (x_1 - m) * torch.exp(-logs) * x_mask
            logdet = None
        else:
            z_1 = (m + torch.exp(logs) * x_1) * x_mask
            logdet = torch.sum(logs * x_mask, [1, 2])

        z = torch.cat([z_0, z_1], 1)
        return z, logdet

    def store_inverse(self):
        self.wn.remove_weight_norm()


class ActNorm(nn.Module):
    def __init__(self, channels, ddi=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(
                device=x.device, dtype=x.dtype)
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len  # [b]

        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m ** 2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*
                                                     self.bias.shape).to(dtype=self.bias.dtype)
            logs_init = (-logs).view(*
                                     self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):
    def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
        super().__init__()
        assert(n_split % 2 == 0)
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian

        w_init = torch.qr(torch.FloatTensor(
            self.n_split, self.n_split).normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert(c % self.n_split == 0)
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(
            b, self.n_split, c // self.n_split, t)

        if reverse:
            if hasattr(self, "weight_inv"):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(
                    dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * \
                    (c / self.n_split) * x_len  # [b]

        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)

        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def store_inverse(self):
        self.weight_inv = torch.inverse(
            self.weight.float()).to(dtype=self.weight.dtype)

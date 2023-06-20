import torch
import torch.nn as nn
import torch.nn.functional as F

class Bilin3d(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3):
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        conv_kw = dict(kernel_size=kernel_size, padding=tuple([k//2 for k in kernel_size]))

        self.conv_in = nn.Conv3d(dim_in, dim_hidden, **conv_kw)
        self.conv_hidden = nn.Conv3d(dim_hidden, dim_hidden, **conv_kw)
        self.bilin_1 = nn.Conv3d(dim_hidden, dim_hidden, **conv_kw)
        self.bilin_21 = nn.Conv3d(dim_hidden, dim_hidden, **conv_kw)
        self.bilin_22 = nn.Conv3d(dim_hidden, dim_hidden, **conv_kw)
        self.conv_out = nn.Conv3d(2 * dim_hidden, dim_in, **conv_kw)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        x = self.conv_out(
            torch.cat([self.bilin_1(x), self.bilin_21(x) * self.bilin_22(x)], dim=1)
        )
        return x

class PriorCost(nn.Module):
    def __init__(self, phi, pre=None, post=None, w=1.):
        super().__init__()
        self.phi = phi
        self.pre = pre if pre is not None else nn.Identity()
        self.post = post if post is not None else nn.Identity()
        self.w = w

    def forward_ae(self, x):
        x = self.pre(x)
        x = self.phi(x)
        x = self.post(x)
        return x

    def forward(self, state):
        return self.w * F.mse_loss(state, self.forward_ae(state))



class MultiPrior(torch.nn.Module):
    def __init__(self, *priors) -> None:
        super().__init__()
        self.priors = torch.nn.ModuleList(priors)

    def forward_ae(self, x):
        return torch.mean(torch.stack([
            prior.forward_ae(x) for prior in self.priors
        ],0), 0)

    def forward(self, x):
        return torch.sum(torch.stack([prior.forward(x) for prior in self.priors]))

class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, pre=None, post=None):
        super().__init__()
        self.pre = pre if pre is not None else nn.Identity()
        self.post = post if post is not None else nn.Identity()
        self._state = []

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        conv_kw = dict(kernel_size=kernel_size, padding=tuple([k//2 for k in kernel_size]))

        self.dim_hidden = dim_hidden
        self.conv_state = torch.nn.Conv3d(dim_in, dim_hidden, bias=False, **conv_kw)
        self.gates = torch.nn.Conv3d( dim_in + dim_hidden, 4 * dim_hidden, **conv_kw)
        self.conv_out = torch.nn.Conv3d( dim_hidden, dim_in, **conv_kw)
        self.dropout = torch.nn.Dropout(dropout)

    def reset_state(self, inp):
        self._grad_norm = None
        # size = [inp.shape[0], self.dim_hidden, *inp.shape[-3:]]
        self._state = [
            self.conv_state(self.pre(torch.zeros_like(inp))),
            self.conv_state(self.pre(torch.zeros_like(inp))),
        ]

    def forward(self, x):
        x = self.pre(x)

        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm
        hidden, cell = self._state
        x = self.dropout(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)

        out = self.post(out)
        return out

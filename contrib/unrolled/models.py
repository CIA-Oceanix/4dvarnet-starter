import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinAe(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out=None, kernel_size=3, downsamp=None):
        super().__init__()
        dim_out = dim_out or dim_in
        self.conv_in = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.bilin_1 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_21 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward(self, x):
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x


class ConvLstmSolver(nn.Module):
    def __init__(self, niter, dim_in, dim_hidden, kernel_size=3, dropout=0.1, ae=None):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.niter = niter
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.ae = ae or (lambda x: x)

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, batch):
        inp = batch.input.nan_to_num()
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        hidden = torch.zeros(size, device=inp.device)
        cell = torch.zeros(size, device=inp.device)
        out = inp
        for _ in range(self.niter):
            hidden = self.dropout(hidden)
            hidden = self.ae(hidden)
            cell = self.dropout(cell)
            out, hidden, cell = self._forward(inp, hidden, cell)
            # out = self.ae(out)
        return out

    def prior_cost(self, x):
        # return F.mse_loss(x, self.ae(x))
        return 0.

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.input.nan_to_num().detach()

    def _forward(self, x, hidden, cell):

        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        out = self.conv_out(hidden)
        return out, hidden, cell

class SimpleConvSolver(nn.Module):
    def __init__(self, niter, dim_in, dim_hidden, kernel_size=3, dropout=0.3, dp_inp=0.0, inp_v='input'):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.niter = niter
        # self.model = torch.nn.Conv2d(dim_in + dim_hidden, 2*dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,)
        # self.model = BilinAe(dim_in + dim_hidden, dim_hidden, 2*dim_hidden, kernel_size=kernel_size)
        self.model = nn.Sequential(
            torch.nn.Conv2d(dim_in + dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,),
            torch.nn.BatchNorm2d(dim_hidden),
            torch.nn.Tanh(),
            torch.nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,),
            torch.nn.BatchNorm2d(dim_hidden),
            torch.nn.Tanh(),
            torch.nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,),
            torch.nn.BatchNorm2d(dim_hidden),
            torch.nn.Tanh(),
            torch.nn.Conv2d(dim_hidden, 2*dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2,),
        )
        self.dropout = torch.nn.Dropout2d(dropout)
        self.dp_inp = torch.nn.Dropout(dp_inp)
        self.conv_out = torch.nn.Conv2d(dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2)
        self.inp_v = inp_v

    def forward(self, batch):
        inp = batch._asdict()[self.inp_v].nan_to_num()
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        state = torch.zeros(size, device=inp.device)
        _inp = self.dropout(inp)
        out = torch.zeros_like(inp)
        for i in range(self.niter):
            state = self.dropout(state)/2 + state/2
            state = self._forward(_inp, state)
        #     out = out /(i+1) + i/(i+1) * self.conv_out(state)
        # if not self.training:
        out = self.conv_out(state)
        return out

    def prior_cost(self, x):
        return 0.

    def init_state(self, batch, x_init=None):
        return 0.

    def _forward(self, x, state):
        inp = (torch.cat((x, state), 1))
        update = self.model(inp)
        a, b = update.chunk(2, 1)
        return a.tanh() * b.sigmoid()

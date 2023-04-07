import numpy as np
import pytorch_lightning as pl
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import einops


class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, opt_fn, norm_stats=None):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight))
        self.test_data = None
        self.norm_stats = norm_stats if norm_stats is not None else (0.0, 1.0)
        self.opt_fn = opt_fn

    @staticmethod
    def weighted_mse(err, weight):
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def forward(self, batch):
        return self.solver(batch)

    def step(self, batch, phase="", opt_idx=None):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)
        grad_loss = self.weighted_mse(
            kornia.filters.sobel(out) - kornia.filters.sobel(batch.tgt), self.rec_weight
        )
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        with torch.no_grad():
            rmse = (
                self.weighted_mse(
                    (out - batch.tgt) * self.norm_stats[1], self.rec_weight
                ) ** 0.5
            )
            self.log(f"{phase}_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(
                f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True
            )

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                batch.input.cpu() * s + m,
                batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )
        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = xr.Dataset(
            {
                k: rec_da.isel(v0=i)
                for i, k in enumerate(["obs", "ssh", "rec_ssh"])
            }
        )


class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.1, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.input.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        state_update = (
            1 / (step + 1) * self.grad_mod(grad)
            + self.lr_grad * (step + 1) / self.n_step * grad
        )
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_ae(state)
        return state


class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
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
        out = self.up(out)
        return out


class BaseObsCost(nn.Module):
    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None):
        super().__init__()
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
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward_ae(self, x):
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        x = self.conv_out(
            torch.cat([self.bilin_1(x), self.bilin_21(x) * self.bilin_21(x)], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state))

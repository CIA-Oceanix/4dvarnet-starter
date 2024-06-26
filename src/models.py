import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils import regrid_interp


class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, test_regrid=None):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.test_regrid = test_regrid
        if test_regrid is not None:
            self.len_dims_regrid = len(test_regrid[0])

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)

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
    
    def regrid_forward(self, batch):

        regrid_out = self(batch)
        reverse_regrid_out = regrid_interp(regrid_out, self.test_regrid)
        return reverse_regrid_out

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def base_step(self, batch, phase=""):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):

        if self.test_regrid is not None:
            return self.test_step_regrid(batch, batch_idx)
        
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

    def test_step_regrid(self, batch, batch_idx):

        batch, original_batch = batch
        if batch_idx == 0:
            self.test_data = []

        m, s = self.norm_stats

        out = self.regrid_forward(batch=batch)

        self.test_data.append(torch.stack(
            [
                original_batch.input.cpu(),
                original_batch.tgt.cpu(),
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'out']

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data,
            self.rec_weight.cpu().numpy()
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data)
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())


class Lit4dVarNetForecast(Lit4dVarNet):
    """
    Lit4dVarNet for forecasting applications:
    solver: function to use as solver
    rec_weight: optimisation weight
    opt_fn: optimisation function
    test_metrics: metrics to run for test
    pre_metric_fn: preprocessing functions to apply to the reconstruction
    norm_stats: normalisation stats of data
    persist_rw: if True: rec_weight saved alongside parameters
    """

    def __init__(self, solver, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, test_regrid=None):
        super().__init__(solver, rec_weight, opt_fn, test_metrics, pre_metric_fn, norm_stats, persist_rw, test_regrid)

    @staticmethod
    def mask_batch(batch):
        new_input = batch.input
        dims = new_input.size()
        new_input[:, dims[1]//2:, :, :] = 0.
        mask_batch = batch._replace(input=new_input)
        mask_batch = batch._replace(input=(batch.input).nan_to_num())
        mask_batch = batch._replace(tgt=(batch.tgt).nan_to_num())
        return mask_batch

    def training_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)
        return super().training_step(mask_batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)
        return super().validation_step(mask_batch, batch_idx)

    def test_step(self, batch, batch_idx):
        if self.test_regrid is not None:
            mask_batch = (self.mask_batch(batch[0]), batch[1])
        else:
            mask_batch = self.mask_batch(batch)
        super().test_step(mask_batch, batch_idx)

    def on_test_epoch_end(self):
        dims = self.rec_weight.size()
        dT = dims[0]
        metrics = []
        for i in range(-((dT - 1) // 2 - 1), 7):
            forecast_weight = np.concatenate(
                (np.zeros((dT // 2 + i, dims[1], dims[2])),
                 np.ones((1, dims[1], dims[2])),
                 np.zeros((dT // 2 - i, dims[1], dims[2]))),
                axis=0)
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data, forecast_weight
            )

            if isinstance(rec_da, list):
                rec_da = rec_da[0]

            test_data_leadtime = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')

            if self.logger:
                test_data_leadtime.to_netcdf(Path(self.logger.log_dir) / f'test_data_{i+(dT-1)//2-1}.nc')
                print(Path(self.trainer.log_dir) / f'test_data_{i+(dT-1)//2-1}.nc')
                

            metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
            metrics_leadtime = pd.Series({
                metric_n: metric_fn(metric_data)
                for metric_n, metric_fn in self.metrics.items()
            })
            metrics.append(metrics_leadtime)

        print(pd.DataFrame(metrics, range(-((dT - 1) // 2 - 1), 7)).T.to_markdown())



class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs):
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

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
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


class GradSolverZero(GradSolver):
    """
    Implementation of the GradSolver with an initialisation at 0, instead of the observations
    """

    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad, **kwargs)

    def init_state(self, batch, x_init=None):
        """
        if x_init is not None : return x_init
        else : return 0
        """
        if x_init is not None:
            return x_init
        return torch.zeros_like(batch.input).requires_grad_(True)


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
        x = x / self._grad_norm
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
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w = w

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return self.w * F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True):
        super().__init__()
        self.bilin_quad = bilin_quad
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

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state))

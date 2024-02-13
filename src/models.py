import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F


class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, opt_fn, sampling_rate = 1, test_metrics=None, pre_metric_fn=None, norm_stats=None, norm_type ='z_score', persist_rw=True):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.sampling_rate = sampling_rate
        self.norm_type = norm_type
        #self.mask = (torch.rand(1, *input_shape) > self.sampling_rate).to('cuda:0')
        print(sampling_rate)

    
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

    @staticmethod
    def weighted_mse_mask(err, weight, mask_nan):
        err_valid = err * mask_nan[None, ...]
        # Calculate the number of valid elements
        num_valid = mask_nan.sum()
    
        if num_valid == 0:
            return torch.tensor(1000.0, device=err.device, requires_grad=True)
        weight_valid = weight * mask_nan[None, ...]

        err_w = err_valid * weight_valid[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        err_w_res = err_w.reshape(err_num.size())
        loss = F.mse_loss(err_w_res[err_num], torch.zeros_like(err_w_res[err_num]))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def forward(self, batch):
        if self.solver.n_step > 0:
            return self.solver(batch)
        else:
            #print(batch.input)
            return self.solver.prior_cost.forward_ae(batch.input)
    
    def step(self, batch, phase=""):
        #print('hello')
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        # Create a mask selecting non-NaN values
        # if self.mask_sampling_with_nan is not None:
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
        # Apply the mask to the input data, setting selected values to NaN
        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')
        batch = batch._replace(input = masked_input)
        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        # nan_count = torch.isnan(batch.input).sum().item()
        # print(f"Number of NaN values: {nan_count}")
        
        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
        training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss
        self.log('sampling_rate', self.sampling_rate, on_step=False, on_epoch=True)
        self.log('weight loss', 10., on_step=False, on_epoch=True)
        self.log('prior cost', 20.,on_step=False, on_epoch=True)
        self.log('grad loss', 5., on_step=False, on_epoch=True)
        #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def base_step(self, batch, phase=""):
        nan_count = torch.isnan(batch.input).sum().item()
        #print(f"Number of NaN values 1: {nan_count}")
        # batch = batch._replace(input = batch.input / torch.bernoulli(torch.full(batch.input.size(), self.sampling_rate)).to('cuda:0'))
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse",  loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        batch_input_clone = batch.input.clone()
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')

        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')
        batch = batch._replace(input = masked_input)
        out = self(batch=batch)
        nan_count = torch.isnan(batch.input).sum().item()
        print(f"Number of NaN values 1: {nan_count}")

        if self.norm_type == 'z_score':
            m, s = self.norm_stats
            self.test_data.append(torch.stack(
                [   batch_input_clone.cpu() * s + m,
                    batch.input.cpu() * s + m,
                    batch.tgt.cpu() * s + m,
                    out.squeeze(dim=-1).detach().cpu() * s + m,
                ],
                dim=1,
            ))

        if self.norm_type == 'min_max':
            min_value, max_value = self.norm_stats
            self.test_data.append(torch.stack(
                [   (batch_input_clone.cpu()  - min_value) / (max_value - min_value),
                    (batch.input.cpu()  - min_value) / (max_value - min_value),
                    (batch.tgt.cpu()  - min_value) / (max_value - min_value),
                    (out.squeeze(dim=-1).detach().cpu()  - min_value) / (max_value - min_value),
                ],
                dim=1,
            ))

    @property
    def test_quantities(self):
        return ['input', 'inp', 'tgt', 'out']

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
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

#class Lit4dVarNet_AE(pl.LightningModule):
#    def __init__(self, prior_cost, rec_weight, opt_fn, sampling_rate = 0.1, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True):
#        super().__init__()
#        self.prior_cost = prior_cost
#        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
#        self.test_data = None
#        self._norm_stats = norm_stats
#        self.opt_fn = opt_fn
#        self.metrics = test_metrics or {}
#        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
#
#        self.sampling_rate = sampling_rate
#        self.mask_sampling = torch.bernoulli(torch.full([4,15,240,240], self.sampling_rate)).to('cuda:0')
#        self.mask_sampling_with_nan = torch.where(self.mask_sampling == 0, float('nan'), self.mask_sampling)
#        print(sampling_rate)
#    
#    @property
#    def norm_stats(self):
#        if self._norm_stats is not None:
#            return self._norm_stats
#        elif self.trainer.datamodule is not None:
#            return self.trainer.datamodule.norm_stats()
#        return (0., 1.)
#
#    @staticmethod
#    def weighted_mse(err, weight):
#        err_w = err * weight[None, ...]
#        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
#        err_num = err.isfinite() & ~non_zeros
#        if err_num.sum() == 0:
#            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
#        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
#        return loss
#
#    def training_step(self, batch, batch_idx):
#        return self.step(batch, "train")[0]
#
#    def validation_step(self, batch, batch_idx):
#        return self.step(batch, "val")[0]
#
#    def forward(self, batch):
#        return self.prior_cost(self.init_state(batch))
#    
#    def step(self, batch, phase=""):
#        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
#            return None, None
#        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
#    
#        # Apply the mask to the input data, setting selected values to NaN
#        masked_input = batch.input.clone()
#        masked_input[mask] = float('nan')
#        batch = batch._replace(input = masked_input)
#
#        autoencoder_loss = self.prior_cost(self.init_state(batch))  
#        self.log(f"{phase}_autoencoder_loss", autoencoder_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log( f"{phase}_gloss", autoencoder_loss, prog_bar=True, on_step=False, on_epoch=True)
#        self.log(f"{phase}_mse", 10000 * autoencoder_loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
#        self.log(f"{phase}_loss", autoencoder_loss, prog_bar=True, on_step=False, on_epoch=True)
#
#        training_loss = autoencoder_loss
#        #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
#        return training_loss, batch.input
#
#    def configure_optimizers(self):
#        return self.opt_fn(self)
#
#    def test_step(self, batch, batch_idx):
#        if batch_idx == 0:
#            self.test_data = []
#        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
#
#        masked_input = batch.input.clone()
#        masked_input[mask] = float('nan')
#        batch = batch._replace(input = masked_input)
#        out = self.prior_cost.forward_ae(self.init_state(batch))
#        m, s = self.norm_stats
#        #print(out.size())
#        self.test_data.append(torch.stack(
#            [
#                batch.input.cpu() * s + m,
#                batch.tgt.cpu() * s + m,
#                out.squeeze(dim=-1).detach().cpu() * s + m,
#            ],
#            dim=1,
#        ))
#
#    def init_state(self, batch, x_init=None):
#        if x_init is not None:
#            return x_init
#
#        return batch.input.nan_to_num().detach().requires_grad_(True)
#    
#    @property
#    def test_quantities(self):
#        return ['inp', 'tgt', 'out']
#
#    def on_test_epoch_end(self):
#        print('hellohellohellohellohello')
#        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
#            self.test_data, self.rec_weight.cpu().numpy()
#        )
#
#        if isinstance(rec_da, list):
#            rec_da = rec_da[0]
#
#        self.test_data = rec_da.assign_coords(
#            dict(v0=self.test_quantities)
#        ).to_dataset(dim='v0')
#
#        metric_data = self.test_data.pipe(self.pre_metric_fn)
#        metrics = pd.Series({
#            metric_n: metric_fn(metric_data) 
#            for metric_n, metric_fn in self.metrics.items()
#        })
#
#        print(metrics.to_frame(name="Metrics").to_markdown())
#        if self.logger:
#            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
#            print(Path(self.trainer.log_dir) / 'test_data.nc')
#            self.logger.log_metrics(metrics.to_dict())
#
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
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

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
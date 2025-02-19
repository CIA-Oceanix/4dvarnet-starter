from src.models import Lit4dVarNet, GradSolverZero, BilinAEPriorCost, BaseObsCost, ConvLstmGradModel
from contrib.multivar.multivar_utils import MultivarBatchSelector
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

import kornia.filters as kfilts

class Multivar4dVarNet(Lit4dVarNet):

    def __init__(self, multivar_selector: MultivarBatchSelector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multivar_selector = multivar_selector
        self._output_norm_stats = None
        self._input_norm_stats = None

    @staticmethod
    def weighted_mse(err, weight):
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            print('ERROR HAS NO FINITE VALUES')
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss
    
    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.placeholder_norm_stats()
        return (0., 1.)

    @property
    def output_norm_stats(self):
        if self._output_norm_stats is not None:
            return self._output_norm_stats
        elif self.trainer.datamodule is not None:
            self._output_norm_stats = self.trainer.datamodule.output_norm_stats()
            return self._output_norm_stats
        return (0., 1.)

    @property
    def input_norm_stats(self):
        if self._input_norm_stats is not None:
            return self._input_norm_stats
        elif self.trainer.datamodule is not None:
            self._input_norm_stats = self.trainer.datamodule.input_norm_stats()
            return self._input_norm_stats
        return (0., 1.)

    def skip_batch(self, batch):
        return self.multivar_selector.multivar_full_output(batch).isfinite().float().mean() < 0.1

    def step(self, batch, phase=""):
        # SKIP BATCH TO IMPLEMENT #
        if self.training and self.skip_batch(batch):
            return None, None

        loss, out = self.multivar_step(batch, phase)
        grad_loss = self.weighted_mse(kfilts.sobel(out) - kfilts.sobel(self.multivar_selector.multivar_full_output(batch)), self.rec_weight)
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out), batch)
        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out
    
    def multivar_step(self, batch, phase=""):
        out = self(batch=batch)
        loss = self.weighted_mse(out - self.multivar_selector.multivar_full_output(batch), self.rec_weight)
        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out
    
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch)
        m, s = self.output_norm_stats

        self.test_data.append(torch.stack(
            [
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

class Multivar4dVarNetForecast(Multivar4dVarNet):
    def __init__(
            self,
            *args,
            rec_weight_fn,
            output_leadtime_start=None,
            output_only_forecast=True,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = output_only_forecast

    @property
    def test_quantities(self):
        return ['out']

    def clear_gpu_mem(self):
        del self.solver
        torch.cuda.empty_cache()

    def mask_batch(self, batch):
        return self.multivar_selector.mask_batch(batch)
    
    def training_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)
        return super().training_step(mask_batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)
        return super().validation_step(mask_batch, batch_idx)

    def test_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)
        super().test_step(mask_batch, batch_idx)

    def get_dT(self):
        return self.rec_weight.size()[0]
    
    def on_test_epoch_end(self):
        # test_data as gpu tensor
        self.clear_gpu_mem()
        print('TEST DATA SIZE: {}'.format(torch.cat(self.test_data).size()))
        #self.test_data = torch.cat(self.test_data).cuda()
        n_output_dims = self.test_data[0].shape[1]

        for output_dim in range(n_output_dims):
            dims = self.rec_weight.size()
            dT = self.get_dT()
            metrics = []
            output_start = 0 if self.output_only_forecast else -((dT - 1) // 2)
            if self.output_leadtime_start is not None:
                output_start = self.output_leadtime_start
            for i in range(output_start, 7):
                leadtime_idx = dT // 2 + i
                forecast_weight = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())[leadtime_idx]
                rec_da = self.trainer.test_dataloaders.dataset.reconstruct_from_items(
                    torch.cat(self.test_data).index_select(dim=2, index=torch.Tensor([leadtime_idx]).type(torch.int64)).index_select(dim=1, index=torch.Tensor([output_dim]).type(torch.int64)).cuda(),
                    forecast_weight,
                    leadtime=leadtime_idx
                )

                if isinstance(rec_da, list):
                    rec_da = rec_da[0]

                test_data_leadtime = rec_da.assign_coords(
                    dict(v0=self.test_quantities)
                ).to_dataset(dim='v'+str(output_dim))

                if self.logger:
                    test_data_leadtime.to_netcdf(Path(self.logger.log_dir) / f'test_data_{i+(dT-1)//2}_ dim{output_dim}.nc')
                    print(Path(self.trainer.log_dir) / f'test_data_{i+(dT-1)//2}_dim{output_dim}.nc')
                    
                metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
                metrics_leadtime = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                metrics.append(metrics_leadtime)

            print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())

class MultivarGradSolverZero(GradSolverZero):

    def __init__(self, multivar_selector: MultivarBatchSelector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multivar_selector = multivar_selector

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return torch.zeros_like(self.multivar_selector.multivar_full_output(batch)).requires_grad_(True)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state, batch) + self.obs_cost(state, batch)
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
            self.grad_mod.reset_state(self.multivar_selector.multivar_full_output(batch))

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_ae(state, batch)
        return state
    
class MultivarBaseObsCost(BaseObsCost):

    def __init__(self, multivar_selector: MultivarBatchSelector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multivar_selector = multivar_selector

    def forward(self, state, batch):
        batch_obs = self.multivar_selector.multivar_obs_input(batch)
        msk = batch_obs.isfinite()
        return self.w * F.mse_loss(self.multivar_selector.multivar_state_obs(state)[msk], batch_obs[msk])

class MultivarBilinAEPriorCost(BilinAEPriorCost):

    def __init__(self, dim_out, dim_hidden, multivar_selector: MultivarBatchSelector, *args, kernel_size=3, **kwargs):
        super().__init__(*args, dim_hidden=dim_hidden, kernel_size=kernel_size, **kwargs)

        self.conv_out = torch.nn.Conv2d(
            2 * dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.multivar_selector = multivar_selector

    def forward_ae(self, x, batch):
        x = torch.concat((x, self.multivar_selector.multivar_prior_input(batch).nan_to_num()), dim=1)
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x
    
    def forward(self, state, batch):
        return F.mse_loss(state, self.forward_ae(state, batch))
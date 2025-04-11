import pandas as pd
import numpy as np
from pathlib import Path
import torch

from src.models import Lit4dVarNetForecast, GradSolverZero, BilinAEPriorCost, GradSolver


class Plus4dVarNetForecast(Lit4dVarNetForecast):
    """
        slight modifications of the Lit4dVarNetForecast model

        rec_weight_fn: function to create alternative reconstruction weights
    """
    def __init__(
            self,
            *args,
            rec_weight_fn,
            output_leadtime_start=None,
            output_leadtime_end=None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_leadtime_end = output_leadtime_end

    def get_dT(self):
        return self.rec_weight.size()[0]

    def on_test_epoch_end(self):
        dims = self.rec_weight.size()
        dT = self.get_dT()
        metrics = []
        output_start = 0 if self.output_only_forecast else -((dT - 1) // 2)
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start
        if self.output_leadtime_end is not None:
            output_end = self.output_leadtime_end
        else:
            output_end = 7
        for i in range(output_start, output_end):
            forecast_weight = self.rec_weight_fn(i, dT, dims, self.rec_weight.cpu().numpy())
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data, forecast_weight
            )

            if isinstance(rec_da, list):
                rec_da = rec_da[0]

            test_data_leadtime = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')

            if self.logger:
                test_data_leadtime.to_netcdf(Path(self.logger.log_dir) / f'test_data_{i+(dT-1)//2}.nc')
                print(Path(self.trainer.log_dir) / f'test_data_{i+(dT-1)//2}.nc')
                

            metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
            metrics_leadtime = pd.Series({
                metric_n: metric_fn(metric_data)
                for metric_n, metric_fn in self.metrics.items()
            })
            metrics.append(metrics_leadtime)

        print(pd.DataFrame(metrics, range(output_start, output_end)).T.to_markdown())

class Plus4dVarNetForecastPatchGPU(Plus4dVarNetForecast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_quantities(self):
        return ['out']

    def clear_gpu_mem(self):
        del self.solver
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        # test_data as gpu tensor
        self.clear_gpu_mem()
        self.test_data = torch.cat(self.test_data).cuda()
        super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)

        if batch_idx == 0:
            self.test_data = []
        out = self(batch=mask_batch)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                #mask_batch.input.cpu() * s + m,
                #mask_batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

# CCut

class Plus4dVarNetForecastPatchGPUCCut(Plus4dVarNetForecastPatchGPU):
    def __init__(self, *args, input_ccut, output_ccut, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_ccut = input_ccut
        self.output_ccut = output_ccut

    def get_dT(self):
        return self.input_ccut*2+1

    def mask_batch(self, batch):

        # temporal masking
        new_input = batch.input[:, :self.input_ccut, :, :]
        new_tgt = batch.tgt[:, :self.output_ccut, :, :]

        mask_batch = batch._replace(input=new_input)
        mask_batch = mask_batch._replace(tgt=new_tgt)

        del batch
        return mask_batch

class GradSolverZeroCCut(GradSolverZero):
    """
    Implementation of the GradSolver with an initialisation at 0, instead of the observations
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_state(self, batch, x_init=None):
        """
        if x_init is not None : return x_init
        else : return 0
        """
        if x_init is not None:
            return x_init
        return torch.zeros_like(batch.tgt).requires_grad_(True)
    
import torch.nn.functional as F

class BilinAEPriorCostCCut(BilinAEPriorCost):
    def __init__(self, *args, dim_in, dim_out, **kwargs):
        self.dim_in_ccut = dim_in
        self.padding = (
            #pad left/right 4th dim (lon)
            0, 0,
            #pad left/right 3rd dim (lat)
            0, 0,
            #pad left/right 2nd dim (channels)
            0, dim_out - dim_in
        )
        super().__init__(*args, dim_in=dim_out, **kwargs)

    def forward_ae(self, x):
        x = F.pad(x, pad=self.padding)
        x = super().forward_ae(x)
        return x[:, :self.dim_in_ccut, :, :]

from torch import nn

class BaseObsCostCCut(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w = w

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return self.w * F.mse_loss(state[:,:batch.input.size(dim=1)][msk], batch.input.nan_to_num()[msk])
    
class GradSolverMDT(GradSolver):
    """
    Implementation of the GradSolver with an initialisation at 0, instead of the observations
    """

    def __init__(self, datamodule, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = datamodule

    def init_state(self, batch, x_init=None):
        """
        if x_init is not None : return x_init
        else : return 0
        """
        if x_init is not None:
            return x_init
        return self.dm.get_mdt().requires_grad_(True)

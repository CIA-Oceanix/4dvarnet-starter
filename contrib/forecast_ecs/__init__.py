import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import src.data
import src.models
import src.utils
import kornia.filters as kfilts
import pandas as pd
import tqdm
import numpy as np
import torch.nn.functional as F
from pathlib import Path

class Lit4dVarNetForecast(src.models.Lit4dVarNet):
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
    def __init__(self, solver, rec_weight, opt_fn,  sampling_rate = 1, test_metrics=None, pre_metric_fn=None, norm_stats=None, norm_type ='z_score', persist_rw=True):
        super().__init__(solver, rec_weight, opt_fn, sampling_rate, test_metrics, pre_metric_fn, norm_stats, norm_type, persist_rw)

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

    #def mask_batch(batch):
    #    new_input = batch.input
    #    dims = new_input.size()
    #    new_input[:, dims[1] - 3:, :, :] = 0.
    #    mask_batch = batch._replace(input=new_input)
    #    mask_batch = batch._replace(input=(batch.input).nan_to_num())
    #    mask_batch = batch._replace(tgt=(batch.tgt).nan_to_num())
    #    return mask_batch
#
    #def training_step(self, batch, batch_idx):
    #    mask_batch = self.mask_batch(batch)
    #    return super().training_step(mask_batch, batch_idx)
#
    #def validation_step(self, batch, batch_idx):
    #    mask_batch = self.mask_batch(batch)
    #    return super().validation_step(mask_batch, batch_idx)
#
    #def test_step(self, batch, batch_idx):
    #    mask_batch = self.mask_batch(batch)
    #    super().test_step(mask_batch, batch_idx)
#
    #def on_test_epoch_end(self):
    #    dims = self.rec_weight.size()
    #    dT = dims[0]
    #    metrics = []
    #    # Adjust the range to ensure the loop allows for a 3 days centered forecast
    #    for i in range(-((dT - 3) // 2), ((dT - 3) // 2) + 1):
    #        # Adjust forecast_weight to focus on 3 consecutive days
    #        forecast_weight = np.concatenate(
    #            (np.zeros((dT // 2 + i - 1, dims[1], dims[2])),  # Adjust for 3 days
    #             np.ones((3, dims[1], dims[2])),  # 3 days forecast
    #             np.zeros((dT // 2 - i - 1, dims[1], dims[2]))),  # Adjust for 3 days
    #             axis=0)
    #        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
    #            self.test_data, forecast_weight
    #        )

    #        if isinstance(rec_da, list):
    #            rec_da = rec_da[0]
#
    #        test_data_leadtime = rec_da.assign_coords(
    #            dict(v0=self.test_quantities)
    #        ).to_dataset(dim='v0')
#
    #        if self.logger:
    #            test_data_leadtime.to_netcdf(Path(self.logger.log_dir) / f'test_data_{i+(dT-3)//2}.nc')
    #            print(Path(self.trainer.log_dir) / f'test_data_{i+(dT-3)//2}.nc')
#
    #        metric_data = test_data_leadtime.pipe(self.pre_metric_fn)
    #        metrics_leadtimFe = pd.Series({
    #            metric_n: metric_fn(metric_data)
    #            for metric_n, metric_fn in self.metrics.items()
    #        })
    #        metrics.append(metrics_leadtime)
#
    #    print(pd.DataFrame(metrics, range(-((dT - 3) // 2), ((dT - 3) // 2) + 1)).T.to_markdown())
#

class GradSolverZero(src.models.GradSolver):
    """
    Implementation of the GradSolver with an initialisation at 0, instead of the observations
    """

    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, weight_obs = 1., weight_prior = 1., lr_grad=0.2, **kwargs):
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, weight_obs, weight_prior, lr_grad, **kwargs)

    def init_state(self, batch, x_init=None):
        """
        if x_init is not None : return x_init
        else : return 0
        """
        if x_init is not None:
            return x_init
        return torch.zeros_like(batch.input).requires_grad_(True)
    

def get_forecast_wei(patch_dims, offset=0, **crop_kw):
    """
    return weight for forecast reconstruction:
    patch_dims: dimension of the patches used
    linear from 0 to 1 where there are obs
    linear from 1 to 0.5 for 7 days of forecast
    0 elsewhere
    """
    pw = src.utils.get_constant_crop(patch_dims, **crop_kw)
    time_patch_weight = np.concatenate(
        (np.linspace(0, 1, (patch_dims['time'] - 1) // 2),
         np.linspace(1, 0.5, 7),
         np.zeros((patch_dims['time'] + 1) // 2 - 7)),
        axis=0)
    final_patch_weight = time_patch_weight[:, None, None] * pw
    return final_patch_weight


#def get_forecast_wei(patch_dims, offset=0, **crop_kw):
#    """
#    return weight for forecast reconstruction:
#    patch_dims: dimension of the patches used
#    linear from 0 to 1 where there are obs
#    linear from 1 to 0.5 for up to the last 3 days of forecast
#    0 for the last three days
#    """
#    pw = src.utils.get_constant_crop(patch_dims, **crop_kw)
#    time_patch_weight = np.concatenate(
#        (np.linspace(0, 1, (patch_dims['time'] - 1) // 2),
#         np.linspace(1, 0.5, (patch_dims['time'] + 1) // 2 - 3),
#         np.zeros(3)),
#        axis=0)
#    final_patch_weight = time_patch_weight[:, None, None] * pw
#    return final_patch_weight
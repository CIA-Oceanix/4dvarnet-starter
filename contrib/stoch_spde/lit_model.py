import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, optim_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, frcst_lead=None, reload_from_maxime=True):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.register_buffer('optim_weight', torch.from_numpy(optim_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.frcst_lead = frcst_lead

        if reload_from_maxime==True:
            if self.frcst_lead is None:
                ckpt = torch.load('/homes/m19beauc/4dvarnet-starter/ckpt/ckpt_spde_wonll_rzf=2.pth')
            elif self.frcst_lead==0:
                ckpt = torch.load('/homes/m19beauc/4dvarnet-starter/ckpt/ckpt_spde_wonll_rzf=2_nrt.pth')
            else:
                ckpt = torch.load('/homes/m19beauc/4dvarnet-starter/ckpt/ckpt_spde_wonll_rzf=2_frcst'+str(self.frcst_lead)+'.pth')
            self.solver.load_state_dict(ckpt)

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
        if (self.frcst_lead is not None) and (self.frcst_lead>0):
            new_input = batch.input
            new_input[:,(-self.frcst_lead):,:,:] = 0.
            batch = batch._replace(input=new_input)
            batch = batch._replace(input=(batch.input).nan_to_num())
            batch = batch._replace(tgt=(batch.tgt).nan_to_num())
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        if (self.frcst_lead is not None) and (self.frcst_lead>0):
            new_input = batch.input
            new_input[:,(-self.frcst_lead):,:,:] = 0.
            batch = batch._replace(input=new_input)
            batch = batch._replace(input=(batch.input).nan_to_num())
            batch = batch._replace(tgt=(batch.tgt).nan_to_num())
        return self.step(batch, "val")[0]

    def forward(self, batch):
        return self.solver(batch)
    
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.optim_weight)
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def base_step(self, batch, phase=""):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.optim_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        if (self.frcst_lead is not None) and (self.frcst_lead>0):
            new_input = batch.input
            new_input[:,(-self.frcst_lead):,:,:] = 0.
            batch = batch._replace(input=new_input)
            batch = batch._replace(input=(batch.input).nan_to_num())
            batch = batch._replace(tgt=(batch.tgt).nan_to_num())

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

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'out']

    def on_test_epoch_end(self):

        if isinstance(self.trainer.test_dataloaders,list):
            rec_da = self.trainer.test_dataloaders[0].dataset.reconstruct(
                self.test_data, self.rec_weight.cpu().numpy()
            )
        else:
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data, self.rec_weight.cpu().numpy()
            )

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

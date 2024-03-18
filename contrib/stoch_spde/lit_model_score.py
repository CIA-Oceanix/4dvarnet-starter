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
    def __init__(self, solver, rec_weight, opt_fn, loss_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, n_simu=2):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.loss_fn = loss_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.n_simu = n_simu

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

    def forward(self, batch, use_conditioning=True):
        return self.solver(batch, use_conditioning)

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        loss, out = self.base_step(batch, phase)
        # MSE grad loss
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        # score loss
        score_loss = self.loss_fn(batch.tgt)
        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log( f"{phase}_score_loss", score_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 10 * loss + 1e-5*score_loss  #+ 200 * grad_loss
        return training_loss, out

    def base_step(self, batch, phase=""):
        out = self(batch=batch)

        # mse loss
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []

        for i in range(self.n_simu):
            self.test_data.append([])

            out = self(batch=batch, use_conditioning=False)
            m, s = self.norm_stats

            self.test_data[i].append(torch.stack(
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

        rec_da_wsimu = []
        for i in range(self.n_simu):
            # reconstruct data
            if isinstance(self.trainer.test_dataloaders,list):
                rec_da = self.trainer.test_dataloaders[0].dataset.reconstruct(
                    self.test_data[i], self.rec_weight.cpu().numpy()
                )
            else:
                rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                    self.test_data[i], self.rec_weight.cpu().numpy()
                )
            rec_da = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')
            rec_da_wsimu.append(rec_da)
        
        if len(rec_da_wsimu)>1:
            self.test_data = xr.concat(rec_da_wsimu, pd.Index(np.arange(self.n_simu), name='simu'))
        else:
            self.test_data = rec_da_wsimu[0]

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


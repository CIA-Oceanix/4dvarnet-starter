import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

import pandas as pd
from pathlib import Path

class Unet(pl.LightningModule):

    def __init__(self, dim_in, channel_dims, rec_weight, opt_fn, rec_weight_fn=None, norm_stats=None, test_metrics=None, pre_metric_fn=None, persist_rw=True, output_leadtime_start=None, output_only_forecast=True, batch_selector=None):
        super().__init__()
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)

        self.rec_weight_fn = rec_weight_fn
        self.output_leadtime_start = output_leadtime_start
        self.output_only_forecast = output_only_forecast

        self.max_depth = len(channel_dims) // 3

        self.ups = list()
        self.up_pools = list()
        self.downs = list()
        self.down_pools = list()
        self.residues = list()

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth*3+1],
                        out_channels=channel_dims[depth*3]
                    ),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth*3+2]*2,
                        out_channels=channel_dims[depth*3+1],
                        kernel_size=3
                    )
                )
            )
            self.up_pools.append(
                torch.nn.MaxPool2d(
                    in_channels=channel_dims[depth*3+3],
                    out_channels=channel_dims[depth*3+2],
                    kernel_size=2
                )
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in if depth==0 else channel_dims[depth*3-1],
                        out_channels=channel_dims[depth*3],
                        kernel_size=3
                    ),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth*3],
                        out_channels=channel_dims[depth*3+1],
                        kernel_size=3
                    )
                )
            )
            self.down_pools.append(
                torch.nn.ConvTranspose2d(
                    in_channels=channel_dims[depth*3+1],
                    out_channels=channel_dims[depth*3+2],
                    kernel_size=2
                )
            )

    def unet_step(self, x, depth):
        x = self.concat_residue(x)
        x, residue = self.down(x)

        if depth == self.max_depth:
            return self.up(x, depth)
        else:
            self.residues.append(residue)
            return self.up(self.unet_step(x, depth+1), depth)

    def forward(self, x):
        return self.unet_step(x, depth=0)

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

    def down(self, x, depth):
        x = self.downs[depth](x)
        return self.down_pools[depth](x), x

    def up(self, x, depth):
        x = self.ups[depth](x)
        return self.up_pools[depth](x), x

    def concat_residue(self, x):
        if len(self.residues) != 0:
            return torch.concat((x, self.residues.pop(-1)))
        else:
            return x
        
    # PYTORCH LIGHTNING LOGIC

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
    def mask_batch(batch):

        # temporal masking
        new_input = batch.input
        dims = new_input.size()
        new_input[:, dims[1]//2:, :, :] = np.nan

        mask_batch = batch._replace(input=new_input)

        return mask_batch

    def training_step(self, batch, batch_idx):
        batch = self.mask_batch(batch)
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        batch = self.mask_batch(batch)
        return self.step(batch, "val")[0]

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None

        out = self(x=batch)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)
        self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out
    
    def configure_optimizers(self):
        return self.opt_fn(self)
    
    @property
    def test_quantities(self):
        return ['out']

    def clear_gpu_mem(self):
        del self.solver
        torch.cuda.empty_cache()

    def test_step(self, batch, batch_idx):
        mask_batch = self.mask_batch(batch)

        if batch_idx == 0:
            self.test_data = []
        out = self(batch=mask_batch)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

    def get_dT(self):
        return self.rec_weight.size()[0]

    def on_test_epoch_end(self):
        self.clear_gpu_mem()
        self.test_data = torch.cat(self.test_data).cuda()

        dims = self.rec_weight.size()
        dT = self.get_dT()
        metrics = []
        output_start = 0 if self.output_only_forecast else -((dT - 1) // 2)
        if self.output_leadtime_start is not None:
            output_start = self.output_leadtime_start
        for i in range(output_start, 7):
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

        print(pd.DataFrame(metrics, range(output_start, 7)).T.to_markdown())
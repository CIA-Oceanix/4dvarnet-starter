import src.models
import xarray as xr
import numpy as np
import kornia.filters as kfilts
import torch
import einops
import contrib.lorenz63



class RearrangedBilinAEPriorCost(src.models.BilinAEPriorCost):
    """
    Wrapper around the base prior cost that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, rearrange_from='b c t', rearrange_to='b t c ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

    def forward_ae(self, x):
        x = einops.rearrange(x, self.rearrange_bef)
        x = super().forward_ae(x)
        x = einops.rearrange(x, self.rearrange_aft)
        return x

class MultiPrior(torch.nn.Module):
    def __init__(self, *priors) -> None:
        super().__init__()
        self.priors = torch.nn.ModuleList(priors)

    def forward_ae(self, x):
        return torch.mean(torch.stack([prior.forward_ae(x) for prior in self.priors],0), 0)

    def forward(self, x):
        return torch.sum(torch.stack([prior.forward(x) for prior in self.priors]))


class SolverWithInit(src.models.GradSolver):
    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.init.nan_to_num().detach().requires_grad_(True)

class RearrangedConvLstmGradModel(src.models.ConvLstmGradModel):
    """
    Wrapper around the base grad model that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, rearrange_from='b c t', rearrange_to='b t c ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        super().reset_state(inp)

    def forward(self, x):
        x = einops.rearrange(x, self.rearrange_bef)
        x = super().forward(x)
        x = einops.rearrange(x, self.rearrange_aft)
        return x

class LitLorenz(src.models.Lit4dVarNet):
    def step(self, batch, phase="", opt_idx=None):
        loss, out = super().base_step(batch, phase)
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        grad_loss = self.weighted_mse( kfilts.sobel(out[..., None]).squeeze()
            - kfilts.sobel(batch.tgt[..., None]).squeeze(), self.rec_weight)
        return  loss + prior_cost, out

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )
        m, s = self.trainer.datamodule.norm_stats()
        rec_ds  = (rec_da * s + m).assign_coords(v0=['input', 'tgt', 'out']).to_dataset(dim='v0')
        rec_ds.to_netcdf('tmp/lorenz.nc')
        print(
            xr.Dataset(dict(
                mse=((rec_ds.tgt - rec_ds.out)**2).mean(),
                percent_err=np.mean((rec_ds.tgt - rec_ds.out)**2)/np.mean(rec_ds.tgt**2)
            )).to_array()
            .to_dataframe(name='4dVarNet').to_markdown()
        )


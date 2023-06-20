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
        crop = 20

        test_data = torch.cat([td[..., crop:-crop] for td in self.test_data])
        print('\n\nPATCH MSE', ((test_data[:, 1] - test_data[:, 2])**2).mean(), '\n\n')
        super().on_test_epoch_end()

        test_data = self.pre_metric_fn(self.test_data)
        print(mse(test_data))
        print(percent_err(test_data))


def mse(rec_ds): return ((rec_ds.tgt - rec_ds.out)**2).mean().values.item()
def percent_err(rec_ds): return (np.mean((rec_ds.tgt - rec_ds.out)**2)/np.mean(rec_ds.tgt**2)).mean().values.item()

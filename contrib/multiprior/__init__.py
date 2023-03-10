import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import collections
import src.data
import src.models
import src.utils

MultiPriorTrainingItem = collections.namedtuple(
    "MultiPriorTrainingItem", ["input", "tgt", "lat", "lon"]
)


def load_data_with_lat_lon(path=None, obs_var="five_nadirs", train_domain=None):
    inp = xr.open_dataset(
        "../sla-data-registry/CalData/cal_data_new_errs.nc"
    )[obs_var]
    gt = (
        xr.open_dataset(
            "../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc"
        )
        .ssh.isel(time=slice(0, -1))
        .interp(lat=inp.lat, lon=inp.lon, method="nearest")
    )

    ds =  xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords)
    if train_domain is not None:
        ds = ds.sel(train_domain)
    return xr.Dataset(
        dict(
            input=ds.input,
            tgt=src.utils.remove_nan(ds.tgt),
            latv=ds.lat.broadcast_like(ds.tgt),
            lonv=ds.lon.broadcast_like(ds.tgt),
        ),
        ds.coords,
    ).transpose('time', 'lat', 'lon').to_array()



class MultiPriorDataModule(src.data.BaseDataModule):
    def get_train_range(self, v):
        train_data = self.input_da.sel(self.xrds_kw.get("domain_limits", {})).sel(
            self.domains["train"]
        )
        return train_data[v].min().values.item(), train_data[v].max().values.item()

    def post_fn(self):
        normalize = lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1]
        lat_r = self.get_train_range("lat")
        lon_r = self.get_train_range("lon")
        minmax_scale = lambda l, r: (l - r[0]) / (r[1] - r[0])
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                MultiPriorTrainingItem._make,
                lambda item: item._replace(tgt=normalize(item.tgt)),
                lambda item: item._replace(input=normalize(item.input)),
                lambda item: item._replace(lat=minmax_scale(item.lat, lat_r)),
                lambda item: item._replace(lon=minmax_scale(item.lon, lon_r)),
            ],
        )


class MultiPriorCost(nn.Module):
    def __init__(self, prior_costs, weight_mod_factory):
        super().__init__()
        self.prior_costs = torch.nn.ModuleList(prior_costs)
        self.weight_mods = torch.nn.ModuleList(
            [weight_mod_factory() for _ in prior_costs]
        )

    def forward_ae(self, state):
        x, coords = state
        phi_outs = torch.stack([phi.forward_ae(x) for phi in self.prior_costs], dim=0)
        phi_weis = torch.softmax(
            torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0), dim=0
        )

        phi_weis = torch.stack([wei(coords, i) for i, wei in enumerate(self.weight_mods)], dim=0)
        
        phi_out = ( phi_outs * phi_weis ).sum(0)
        return phi_out

    def forward(self, state):
        return nn.functional.mse_loss(state[0], self.forward_ae(state))

class MultiPriorGradSolver(src.models.GradSolver):
    def init_state(self, batch, x_init=None):
        x_init = super().init_state(batch, x_init)
        coords = torch.stack((batch.lat[:,0], batch.lon[:,0]), dim=1)
        return (x_init, coords)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        x, coords = state
        grad = torch.autograd.grad(var_cost, x, create_graph=True)[0]

        x_update = (
            1 / (step + 1) * self.grad_mod(grad)
            + self.lr_grad * (step + 1) / self.n_step * grad
        )
        state = (x - x_update, coords)
        return state

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = [s.detach().requires_grad_(True) for s in state]

            if not self.training:
                state = [self.prior_cost.forward_ae(state), state[1]]
        return state[0]

class MultiPriorObsCost(nn.Module):
    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return nn.functional.mse_loss(state[0][msk], batch.input.nan_to_num()[msk])

class BinWeightMod(nn.Module):
    def forward(self, x, n_prior):
        if n_prior == 0:
            return torch.ones(x.shape[0], device=x.device)[..., None, None, None]

        else:
            return torch.zeros(x.shape[0], device=x.device)[..., None, None, None]


class WeightMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )


    def forward(self, x, *args, **kwargs):
        x = einops.reduce(x, 'b c lat lon -> b c', 'mean')
        w = self.net(w)
        return w[..., None, None]


def multiprior_train(trainer, model, dm, test_domain):
    """
    Before multiprior
    |          |   osse_metrics |
    |:---------|---------------:|
    | RMSE (m) |      0.0257443 |
    | λx       |      0.924     |
    | λt       |      5.6       |
    | μ        |      0.94336   |
    | σ        |      0.01276   |
    """
    ckpt_path = "/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-02-27/19-18-09/base/checkpoints/val_rmse=0.0153-epoch=286.ckpt"
    ckpt = torch.load(ckpt_path)['state_dict']
    print(model.load_state_dict(
        {k.replace('solver.prior_cost', 'solver.prior_cost.prior_costs.0'): v for k,v in ckpt.items()}
    , strict=False))
    for param in model.solver.prior_cost.prior_costs[0].parameters():
        param.requires_grad = False
    trainer.fit(model, dm)
    src.utils.test_osse(trainer, model, dm, test_domain, None)


if __name__ == "__main__":
    import config
    import torch
    import matplotlib.pyplot as plt
    import hydra
    import contrib.multiprior
    import importlib
    import src.utils
    importlib.reload(contrib.multiprior)
    

    src.utils.load_cfg("outputs/2023-03-08/17-31-55")


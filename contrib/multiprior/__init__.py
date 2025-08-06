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

    # ds =  xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords)
    ds = (
        xr.open_dataset('../sla-data-registry/qdata/natl20.nc')
        # .assign(ssh=lambda ds: ds.ssh.coarsen(lon=2, lat=2).mean().interp(lat=ds.lat, lon=ds.lon))
        .load()
        .assign(
            input=lambda ds: ds.nadir_obs,
            tgt=lambda ds: ds.ssh,
        )
    )
    # if train_domain is not None:
    #     ds = ds.sel(train_domain)
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
        minmax_scale = lambda l, r: 2 * (l - r[0]) / (r[1] - r[0]) - 1.
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
        var_cost = self.prior_cost(state) + self.obs_cost(state[0], batch)
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


class BinWeightMod(nn.Module):
    def forward(self, x, n_prior):
        if n_prior == 0:
            return torch.ones(x.shape[0], device=x.device)[..., None, None, None]

        else:
            return torch.ones(x.shape[0], device=x.device)[..., None, None, None]*float('-inf')


class WeightMod(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsamp = 5
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
        )


    def forward(self, x, *args, **kwargs):
        x = einops.reduce(
            x,
            'b c (rlat lat) ( rlon lon) -> b c lat lon',
            'mean',
            rlat=self.downsamp,
            rlon=self.downsamp,
        )
        x = self.net(x)
        x = nn.functional.interpolate(x, scale_factor=self.downsamp, mode='bilinear')
        return x


def multiprior_train(trainer, model, dm, test_domain):
    print()
    print(trainer.logger.log_dir)
    print()
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
    # ckpt_path = "/raid/localscratch/qfebvre/4dvarnet-starter/outputs/2023-02-27/19-18-09/base/checkpoints/val_rmse=0.0153-epoch=286.ckpt"
    # ckpt = torch.load(ckpt_path)['state_dict']
    # print(model.load_state_dict(
    #     {
    #         k.replace('solver.prior_cost', 'solver.prior_cost.prior_costs.0'): v
    #         for k,v in ckpt.items()
    #         if "prior_cost" in k
    #     }
    # , strict=False))
    # for param in model.solver.prior_cost.prior_costs[0].parameters():
    #     param.requires_grad = False
    trainer.fit(model, dm)
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    src.utils.test_osse(trainer, model, dm, test_domain, best_ckpt_path)


if __name__ == "__main__":
    import config
    import torch
    import matplotlib.pyplot as plt
    import hydra
    import contrib.multiprior
    from tqdm import tqdm
    import importlib
    import src.utils
    import numpy as np
    importlib.reload(contrib.multiprior)
    from pathlib import Path
    import xarray as xr
    
    with hydra.initialize('4dvarnet-starter/config', version_base="1.3"):
        cfg = hydra.compose('main', overrides=['xp=multiprior', 'trainer.logger=False'])
        trainer = hydra.utils.call(cfg.trainer)
        model = hydra.utils.call(cfg.model)
        dm = hydra.utils.call(cfg.datamodule)
        test_domain = hydra.utils.call(cfg.entrypoints[1].test_domain)


    xp_dir = Path("outputs/2023-03-11/17-08-27")

    xp_dir = Path("outputs/2023-03-12/17-37-37")
    xp_dir =Path("outputs/2023-03-13/10-19-05")
    xp_dir =Path("outputs/2023-03-13/16-16-45")
    xp_dir =Path("outputs/2023-03-13/10-19-05")
    # Path("outputs/2023-03-10/18-09-19")
    cfg, xp = src.utils.load_cfg(xp_dir / ".hydra")
    ckpt = src.utils.best_ckpt(xp_dir /xp)
    lit_mod = hydra.utils.call(cfg.model)
    dm = hydra.utils.call(cfg.datamodule)
    test_domain = hydra.utils.call(cfg.entrypoints[1].test_domain)
    lit_mod.load_state_dict(torch.load(ckpt)['state_dict'])
    device = 'cuda:7'
    lit_mod = lit_mod.eval().to(device)
    dm.setup()
    m, s = dm.norm_stats()
    dl = dm.test_dataloader()
    coords = dl.dataset.get_coords()
    coords_it = iter(coords)
    batches = []
    with torch.no_grad():
        for batch in tqdm(dl):
            batch = dm.transfer_batch_to_device(batch, lit_mod.device, 0)
            out = lit_mod(batch)
            state = lit_mod.solver.init_state(batch, out)
            x, c = state
            phi_outs = torch.stack(
                [phi.forward_ae(x).cpu() * s + m for phi in lit_mod.solver.prior_cost.prior_costs], dim=1)
            phi_weis = torch.ones_like(phi_outs) * torch.softmax(
                torch.stack([wei(c, i) for i, wei in enumerate(lit_mod.solver.prior_cost.weight_mods)], dim=1),
            dim=1).cpu()
            [*lit_mod.solver.prior_cost.weight_mods[1].named_parameters()]
            batch_out = (out[:, None, ...].detach().cpu() * s) + m
            batch_tgt = (batch.tgt[:, None, ...].detach().cpu() * s) + m
            batches.append(torch.cat([phi_outs, phi_weis, batch_out, batch_tgt], dim=1))

    test_data = dl.dataset.reconstruct(batches)
    test_ds = xr.Dataset(
        {
            k: test_data.isel(v0=i)
            for i, k in enumerate(["phi_out0", "phi_out1","phi_wei0", "phi_wei1", "out", "gt"])
        }
    )
    test_ds[["phi_wei0", "phi_wei1"]].sel(test_domain).isel(lon=slice(0,200)).isel(time=10).to_array().plot(col='variable', col_wrap=2)
    test_ds.pipe(lambda ds: ds -ds.gt)[["phi_out0", "phi_out1", "out"]].sel(test_domain).isel(lon=slice(0,200)).isel(time=10).to_array().plot(col='variable', col_wrap=2, robust=True)
    plt.imshow(c.cpu()[0,1])
    plt.imshow(phi_weis.cpu()[0,1,0])
    plt.imshow(phi_outs.cpu()[0,0,0])

    test_ds[["phi_wei0", "phi_wei1"]].isel(time=1).to_array().plot(row="variable", figsize=(10,10))
    test_ds[["phi_out0", "phi_out1", "out", "gt"]].isel(time=1).map(src.utils.geo_energy).to_array().plot(row="variable", figsize=(10,10))
    test_ds.isel(time=1).to_array().plot(row="variable", figsize=(10,10))
    test_ds.isel(time=1).phi_out1.pipe(src.utils.geo_energy).plot(figsize=(10,3))
    test_ds.isel(time=1).phi_out0.pipe(src.utils.geo_energy).plot(figsize=(10,3))
    test_ds.isel(time=1).phi_out1.plot(figsize=(10,3))
    test_ds.isel(time=1).out.pipe(src.utils.geo_energy).plot(figsize=(10,3))
    test_ds.isel(time=1).out.pipe(src.utils.geo_energy).plot(figsize=(10,3))
    model.solver.prior_cost.prior_costs

    test_ds[["phi_wei0", "phi_wei1"]].sel(test_domain).mean(('time', 'lat')).to_array().plot(hue='variable', figsize=(10, 4))

import torch
import xarray as xr

def grad_mod_finetune(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": 0},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": 0},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }


def load_glob_2019_tracks(path='../sla-data-registry/glob_tracks_2019/alg_noin_alg.nc'):
    return (
        xr.open_dataset(path, engine='netcdf4')[['others']]
        .rename({'others': 'input'})
        .assign(tgt=lambda ds: ds.input)
        .to_array()
        .sortby('variable')
    )

import numpy as np
import functools as ft
import xarray as xr
from scipy.integrate import solve_ivp
import scipy.interpolate
import collections
import src.data

TrainingItemWithInit = collections.namedtuple(
    "TrainingItemWithInit", sorted(["init", "input", "tgt"])
)

def dyn_lorenz63(t, x, sigma=10., rho=28., beta=8./3):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(x[1]-x[0])
    x_2 = x[0]*(rho-x[2])-x[1]
    x_3 = x[0]*x[1] - beta*x[2]
    dx  = np.array([x_1, x_2, x_3])
    return dx


def trajectory_da(fn, y0, solver_kw, warmup_kw=None):
    if warmup_kw is not None:
        warmup = solve_ivp(fn, y0=y0, **{**solver_kw, **warmup_kw})
        y0 = warmup.y[:,-1]
    warmup = solve_ivp(fn, y0=y0, **solver_kw)
    return xr.DataArray(warmup.y, dims=('component', 'time'), coords={'component': ['x', 'y', 'z'], 'time': warmup.t})


def only_first_obs(da):
    new_da = xr.full_like(da, np.nan)
    new_da.loc['x']=da.loc['x']
    return new_da

def subsample(da, sample_step=20):
    new_da = xr.full_like(da, np.nan)
    new_da.loc[:, ::sample_step]=da.loc[:, ::sample_step]
    return new_da

def add_noise(da, sigma=2**.5):
    return da  + np.random.randn(*da.shape) * sigma

def interpolate_grid_data(npa):
    data_points = np.nonzero(np.isfinite(npa))

    if len(data_points[0]) == 0:
        return npa
    data_values = npa[data_points]
    tgt_points = np.nonzero(np.ones_like(npa))
    tgt_values = scipy.interpolate.griddata(
        points=data_points,
        values=data_values,
        xi=tgt_points,
        method='cubic'
    )

    new_npa = np.zeros_like(npa)
    new_npa[tgt_points] = tgt_values
    return new_npa

def training_da(traj_da, obs_fn):
    return xr.Dataset(
        dict(
            input=traj_da.pipe(obs_fn),
            tgt=traj_da,
    )).assign(
        init=lambda ds: (
                ds.input
                .to_dataset(dim='component')
                .map(lambda da: xr.apply_ufunc(interpolate_grid_data, da))
                .to_array(dim='component')
        )
    ).to_array().sortby('variable')


class LorenzDataModule(src.data.BaseDataModule):
    def post_fn(self):

        normalize = lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1]
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                TrainingItemWithInit._make,
                lambda item: item._replace(tgt=normalize(item.tgt)),
                lambda item: item._replace(input=normalize(item.input)),
                lambda item: item._replace(init=normalize(item.init)),
            ],
        )

if __name__ == '__main__':
    import contrib.lorenz63
    import hydra
    with hydra.initialize(None, None, version_base='1.3'):
        cfg = hydra.compose("base_lorenz")





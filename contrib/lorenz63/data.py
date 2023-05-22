from bokeh.embed import components
import numpy as np
from torch.autograd import variable
import xarray as xr
from scipy.integrate import solve_ivp

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


def obs_only_first(da, sampling_step=20):
    new_da = xr.full_like(da, np.nan)
    new_da.loc['x', ::sampling_step]=da.loc['x', ::sampling_step]
    return new_da

def add_noise(da, sigma=2**.5):
    return da  + np.random.randn(*da.shape) * sigma

def training_da(traj_da, mask_fn, noise_fn):
    return xr.Dataset(
        dict(
            tgt=traj_da,
            input=traj_da.pipe(mask_fn).pipe(noise_fn)
    )).to_array().sortby('variable')


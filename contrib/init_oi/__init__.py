import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.data
import src.models
import src.utils
import kornia.filters as kfilts
import pandas as pd
import tqdm
import numpy as np
from xrpatcher import XRDAPatcher
from functools import partial
    
class Lit4dVarNet_OI(src.models.Lit4dVarNet):
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
        # Apply the mask to the input data, setting selected values to NaN
        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')

        lat_values = np.linspace(-65.95, -54, 240)
        lon_values = np.linspace(32, 43.95, 240)
        time_values = pd.date_range("2009-07-01T12:00:00", periods=15, freq='D')

        # Create a DataArray with zeros
        data = np.zeros((len(time_values), len(lat_values), len(lon_values)))

        # Create the xarray DataArray
        da = xr.DataArray(data, coords=[('time', time_values), ('lat', lat_values), ('lon', lon_values)])

        # If you need a Dataset instead
        ds = da.to_dataset(name='variable_name')

        masked_input_np = masked_input.cpu().numpy()

        # Assuming masked_input has the same shape as 'data' used to create 'ds'
        # and the dimensions are in the order (time, lat, lon)

        # Step 2: Create the xarray DataArray using the same coordinates
        masked_input_da= xr.DataArray(masked_input_np[0], coords=[('time', time_values), ('lat', lat_values), ('lon', lon_values)])

        # Step 3: Optionally convert to Dataset 
        masked_input_ds = masked_input_da.to_dataset(name='ecs')
        patcher_cls= partial(XRDAPatcher,
                    patches=dict(time=5, lat=40, lon=40),
                    strides=dict(time=5, lat=40, lon=40)
                )
        lt = pd.to_timedelta('7D')
        lx = 1.5
        ly = 1.5
        noise = 0.05
        obs_dt = pd.to_timedelta('1D')
        obs_dx = 0.25
        obs_dy = 0.25

        interpolated_da = oi(outgrid_da = xr.DataArray(np.zeros_like(masked_input_da.values), dims=masked_input_da.dims, coords=masked_input_da.coords), patcher_cls= patcher_cls, obs = masked_input_ds, obs_var='ecs', lt=lt, lx=lx, ly=ly, noise=noise, obs_dt=obs_dt, obs_dx=obs_dx, obs_dy=obs_dy, device='cuda:0')
        interpolated_tensor = torch.tensor(interpolated_da.values).to('cuda:0')
        interpolated_tensor = interpolated_tensor.unsqueeze(0)
        batch = batch._replace(input = interpolated_tensor)
        if self.solver.n_step > 0:

            loss, out = self.base_step(batch, phase)
            grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
            prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss
            
            self.log('sampling_rate', self.sampling_rate, on_step=False, on_epoch=True)
            self.log('weight loss', 10., on_step=False, on_epoch=True)
            self.log('prior cost', 20.,on_step=False, on_epoch=True)
            self.log('grad loss', 5., on_step=False, on_epoch=True)
            #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
            return training_loss, out
        
        else:
            loss, out = self.base_step(batch, phase)
            return loss, out
        
    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        batch_input_clone = batch.input.clone()
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
        # Apply the mask to the input data, setting selected values to NaN
        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')
        batch = batch._replace(input = masked_input)
        out = self(batch=batch)

        if self.norm_type == 'z_score':
            m, s = self.norm_stats
            self.test_data.append(torch.stack(
                [   batch_input_clone.cpu() * s + m,
                    batch.input.cpu() * s + m,
                    batch.tgt.cpu() * s + m,
                    out.squeeze(dim=-1).detach().cpu() * s + m,
                ],
                dim=1,
            ))

        if self.norm_type == 'min_max':
            min_value, max_value = self.norm_stats
            self.test_data.append(torch.stack(
                [   (batch_input_clone.cpu()  - min_value) / (max_value - min_value),
                    (batch.input.cpu()  - min_value) / (max_value - min_value),
                    (batch.tgt.cpu()  - min_value) / (max_value - min_value),
                    (out.squeeze(dim=-1).detach().cpu()  - min_value) / (max_value - min_value),
                ],
                dim=1,
            ))

#def oi_pytorch(obs, patch_size, stride, lt=7., lx=1., ly=1., noise=0.05, obs_dt =1., obs_dx = 0.25, obd_dy=0.25):
#    patches = obs.unfold(0, patch_size[0], stride[0])\
#                     .unfold(1, patch_size[1], stride[1])\
#                     .unfold(2, patch_size[2], stride[2])
#    
#    patcher_cls = patches.contiguous().view(-1, *patch_size)
#    
#    for patch in patcher_cls:
#        obss = select_observations(obs, patch, lt, ly, lx)
#        bin_edges = torch.arange(obs.time.min(), obs.time.max() + obs_dt, obs_dt)
#        pobs = bin_and_average(obss, bin_edges)


def oi(outgrid_da,
        patcher_cls,
        obs,
        obs_var='ecs',
        lt=pd.to_timedelta('7D'), lx=1., ly=1.,
        noise=0.05,
        obs_dt=pd.to_timedelta('1D'),
        obs_dx=0.25,
        obs_dy=0.25,
        device='cuda'
    ):

    """
    outgrid_da: xr.DataArray with (time, lat, lon) dims and coords with target region, period and resolution, will be updated inplace
    patcher_cls: callable that return an DataArray iterator of slices of outgrid_da to be sequentially solved for
    obs: xr.Dataset of observations with (time, lat, lon) coords
    obs_var: observation values to be interpolated
    lt: time decorellation factor for covariance model
    lx: longitudinal decorellation factor for covariance model
    ly: latitudinal decorellation factor for covariance model
    noise: noise variance
    obs_dt: temporal bin size for obs coarsening 
    obs_dx: longitudinal bin size for obs coarsening 
    obs_dy: latitudinal bin size for obs coarsening 
    device: torch device to run oi on 'cpu' or 'cuda' gpu
    
    Returns: xr.DataArray outgrid_da filled with interpolation
    """

    patcher = patcher_cls(outgrid_da)
    # 1 Iterate over each patch
    for p in patcher:
        # 2 select observations within 2 stds in each direction
        pobs = obs.where(
            (np.isfinite(obs[obs_var]))
        &  (obs.time >= (p.time.min() - 2 * lt))
        &  (obs.time <= (p.time.max() + 2 * lt))
        &  (obs.lat >= (p.lat.min() - 2 * ly))
        &  (obs.lat <= (p.lat.max() + 2 * ly))
        &  (obs.lon >= (p.lon.min() - 2 * lx))
        &  (obs.lon <= (p.lon.max() + 2 * lx)), drop=True
        )

        # 3 Coarsen the observations to a certain resolution
        df = pobs.to_dataframe().reset_index().dropna()
        tres = pd.to_timedelta('1D')
        xres = 0.25
        time_bins = pd.date_range(df['time'].min(), df['time'].max() + obs_dt, freq=obs_dt)
        lat_bins = np.arange(df['lat'].min(), df['lat'].max() + obs_dx, obs_dx)
        lon_bins = np.arange(df['lon'].min(), df['lon'].max() + obs_dy, obs_dy)

        ## bin
        df['time_bin'] = pd.cut(df['time'], time_bins)
        df['lat_bin'] = pd.cut(df['lat'], lat_bins)
        df['lon_bin'] = pd.cut(df['lon'], lon_bins)

        ## Average data within each bin
        df_averaged = df.groupby(['time_bin', 'lat_bin', 'lon_bin'], observed=True).mean(numeric_only=False).reset_index()
        pobs_coarse = df_averaged.dropna().set_index('time')[[obs_var, 'lat', 'lon']].to_xarray()

        # Create flat coordinates and values array
        obs_values, obs_time, obs_lat, obs_lon = pobs_coarse[obs_var].values, pobs_coarse.time.values, pobs_coarse.lat.values, pobs_coarse.lon.values
        gtime, glat, glon = (np.ravel(x) for x in np.meshgrid(p.time, p.lat, p.lon, indexing='ij'))

        # Convert to torch tensor
        d0 = pd.to_datetime("2010-01-01").to_datetime64()
        _obs_time = (obs_time - d0) / lt
        _gtime = (gtime -d0) / lt
        _lt = lt / lt
        (
            tobs_values, tobs_time, tobs_lat, tobs_lon,
            tgtime, tglat, tglon
        ) = (
            torch.from_numpy(t).to(device).float()
            for t in [
                obs_values, _obs_time, obs_lat, obs_lon,
                _gtime, glat, glon
            ]
        )

        # 4 Solve oi
        BHt = torch.exp(
                - ((tgtime[:, None] - tobs_time[None, :]) / _lt)**2
                - ((tglon[:, None] - tobs_lon[None, :]) / lx)**2
                - ((tglat[:, None] - tobs_lat[None, :]) / ly)**2
            )
        HBHt = torch.exp(-((tobs_time[:, None] - tobs_time[None, :]) / _lt)**2 -
                        ((tobs_lon[:, None] - tobs_lon[None, :]) / lx)**2 -
                        ((tobs_lat[:, None] - tobs_lat[None, :]) / ly)**2)

        nobs = len(tobs_time)
        R = torch.diag(torch.full((nobs,), noise**2, device=tobs_time.device))


        Coo = HBHt + R
        Mi = torch.linalg.inv(Coo)
        Iw = torch.mm(BHt, Mi)
        sol=torch.mv(Iw, tobs_values)

        # fill in outputput da
        p[:] = sol.detach().cpu().numpy().reshape(p.shape)

    return outgrid_da

#def bin_and_average(data, bin_edges):
#    """
#    Bins the data along the first dimension and averages the values within each bin.
#    
#    Parameters:
#    - data: A PyTorch tensor of shape (N, ...) where N is the number of data points.
#    - bin_edges: A PyTorch tensor of shape (M,) representing the edges of M-1 bins.
#    
#    Returns:
#    - A PyTorch tensor containing the averaged data for each bin.
#    """
#    data = data.to(bin_edges.device)
#    num_bins = bin_edges.shape[0] - 1
#    bin_sums = torch.zeros(num_bins, *data.shape[1:], device=data.device)
#    bin_counts = torch.zeros(num_bins, device=data.device)
#    for i in range(num_bins):
#        in_bin = (data[:, 0] >= bin_edges[i]) & (data[:, 0] < bin_edges[i + 1])
#        bin_sums[i] += data[in_bin].sum(dim=0)
#        bin_counts[i] += in_bin.sum()
#    bin_counts[bin_counts == 0] = 1
#    bin_averages = bin_sums / bin_counts.unsqueeze(1)
#    return bin_averages
#
#def select_observations(obs, obs_var, p, lt, ly, lx):
#    valid_obs = torch.isfinite(obs[..., obs_var])
#    time_cond = (obs.time >= (p.time.min() - 2 * lt)) & (obs.time <= (p.time.max() + 2 * lt))
#    lat_cond = (obs.lat >= (p.lat.min() - 2 * ly)) & (obs.lat <= (p.lat.max() + 2 * ly))
#    lon_cond = (obs[0, 0, :, 2] >= (p.lon.min() - 2 * lx)) & (obs[0, 0, :, 2] <= (p.lon.max() + 2 * lx))
#
#    combined_cond = valid_obs & time_cond.unsqueeze(1).unsqueeze(2) & lat_cond.unsqueeze(0).unsqueeze(2) & lon_cond.unsqueeze(0).unsqueeze(1)
#
#    return obs[combined_cond]
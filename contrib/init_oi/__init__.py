import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import src.data
import src.models
import src.utils
import kornia.filters as kfilts
import pandas as pd
import tqdm
import numpy as np
    
class Lit4dVarNet_OI(src.models.Lit4dVarNet):
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        mask = (torch.rand(batch.input.size()) > self.sampling_rate).to('cuda:0')
        # Apply the mask to the input data, setting selected values to NaN
        masked_input = batch.input.clone()
        masked_input[mask] = float('nan')
        batch = batch._replace(input = masked_input)
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
    for p in tqdm.tqdm(patcher):
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
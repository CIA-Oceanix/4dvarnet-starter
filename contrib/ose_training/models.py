import numpy as np
import pandas as pd
from pathlib import Path
import torch
import einops
import src.models
import torch.nn.functional as F


def scale_tgt_coords(batch):
    minmax = lambda t: (
        einops.reduce( t.nan_to_num(nan=torch.inf), 'b ... -> b ()', 'min'),
        einops.reduce( t.nan_to_num(nan=-torch.inf), 'b ... -> b ()', 'max'),
    )
    def rescale(inp, tgt):
        m, M = minmax(inp)
        return (tgt - m) / (M - m) * 2 -1

    return torch.stack([
        rescale(batch.input_coords.lon, batch.tgt_coords.lon),
        rescale(batch.input_coords.lat, batch.tgt_coords.lat),
        rescale(batch.input_coords.time, batch.tgt_coords.time),
    ], -1)

def interp(grid, batch):
    tgt_c = scale_tgt_coords(batch)
    sample = F.grid_sample(
        grid[:, None, ...],
        einops.rearrange(tgt_c, 'b n c -> b () () n c'),
        align_corners=False, mode='bilinear'
    )
    return einops.rearrange(sample, 'b () () () n -> b n')

class LitOse(src.models.Lit4dVarNet):
    def step(self, batch, phase=''):
        out = self(batch)
        # print(out.shape)

        track_out = interp(out, batch) 
        track_w = interp(
            einops.repeat(self.rec_weight, 't y x -> b t y x', b=out.shape[0]),
            batch
        )
        loss = self.weighted_mse(batch.tgt - track_out, track_w)
        gloss = self.weighted_mse(batch.tgt.diff(1, -1) - track_out.diff(1, -1), track_w[..., :-1])
        lloss = self.weighted_mse(batch.tgt.diff(2, -1) - track_out.diff(2, -1), track_w[..., :-2])
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
        self.log( f"{phase}_mse", 10000 * loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=phase=='val')
        self.log( f"{phase}_gloss", gloss, prog_bar=True, on_step=False, on_epoch=True)
        self.log( f"{phase}_lloss", lloss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 100 * gloss + 100 * lloss + prior_cost
        # self.log( f"{phase}_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return training_loss, out

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                batch.input.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

    @property
    def test_quantities(self):
        return ['inp', 'out']

    def on_test_epoch_end(self):
        # import lovely_tensors
        # lovely_tensors.monkey_patch()
        # print(self.test_data)
        #
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data,
            dims_labels=['v0', 'time', 'lat', 'lon'],
            weight=self.rec_weight.cpu().numpy()
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]
        # print(rec_da)
        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())

if __name__ == '__main__':

    import contrib.ose_training.data
    import contrib.ose_training.models
    import importlib
    importlib.reload(contrib.ose_training.models)
    dl_kws = dict(batch_size=2, num_workers=1)
    domain_limits = dict(time=slice('2010-01-01', '2021-01-01'), lat=slice(32, 44), lon=slice(-66, -54))
    patcher_kws = dict(patches=dict(time=15, lat=240, lon=240), strides=dict(time=10), domain_limits=domain_limits)
    ps = list(Path('../sla-data-registry/ose_training').glob('*.nc'))
    train_ds = contrib.ose_training.data.OseDataset(path=ps[0], patcher_kws=patcher_kws)
    dm = contrib.ose_training.data.OseDatamodule(train_ds, train_ds, train_ds, dl_kws=dl_kws)
    dm.setup()
    dl = dm.train_dataloader()
    b = next(iter(dl))


    out = torch.ones_like(b.input)

    tgt_c = scale_tgt_coords(b)
    sample = F.grid_sample(out[:, None, ...], tgt_c[:, None,   None, ...])

    b.tgt

    item = train_ds[0]
    nad_item = train_ds.nad_ds.sel(nad_time=self.time_range(grid_item))
    train_ds.patcher[0]
    duacs = xr.open_dataset('../sla-data-registry/data_OSE/NATL/training/ssh_alg_h2g_j2g_j2n_j3_s3a_duacs.nc')
    tracks = xr.open_dataset('../sla-data-registry/data_OSE/along_track/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc')
    tt = slice('2017-01-01', '2017-01-10')
    gf = dict(lat=slice(32, 44), lon=slice(-65, -55))
    grid = duacs.sel(time=tt).sel(gf)
    nad = (
        tracks.sel(time=tt)
        .rename(longitude='lon', latitude='lat')
        .assign(ssh=lambda ds:ds.mdt+ds.sla_filtered-ds.lwe)[['ssh']]
        .assign_coords(lon=lambda ds: (ds.lon +180) %360 -180)
        .pipe(lambda ds: ds.where(ds.lat > gf['lat'].start, drop=True))
        .pipe(lambda ds: ds.where(ds.lat < gf['lat'].stop, drop=True))
        .pipe(lambda ds: ds.where(ds.lon > gf['lon'].start, drop=True))
        .pipe(lambda ds: ds.where(ds.lon < gf['lon'].stop, drop=True))
    )
    duacs_nad = geogrid.grid_to_coord_based(grid, nad)
    merged = xr.merge([nad, duacs_nad.rename(ssh='duacs')])
    merged.isel(time=slice(0,100)).to_array().plot(hue='variable')

    b = contrib.ose_training.data.TrainingItem(
        input=torch.from_numpy(grid.ssh.transpose('time', 'lat', 'lon').values).float()[None],
        input_coords= contrib.ose_training.data.Coords(
            time=torch.from_numpy((grid.time - grid.time.min()).values / pd.to_timedelta('1D')).float()[None],
            lat=torch.from_numpy((grid.lat).values).float()[None],
            lon=torch.from_numpy((grid.lon).values).float()[None],
        ),
        tgt=torch.from_numpy(nad.ssh.values).float()[None],
        tgt_coords= contrib.ose_training.data.Coords(
            time=torch.from_numpy((nad.time - grid.time.min()).values / pd.to_timedelta('1D')).float()[None],
            lat=torch.from_numpy((nad.lat).values).float()[None],
            lon=torch.from_numpy((nad.lon).values).float()[None],
        )
    )
    b.tgt_coords.time.round()
    b.input_coords.time

    import contrib.ose_training.data
    import contrib.ose_training.models
    import importlib
    importlib.reload(contrib.ose_training.models)
    tinterp = contrib.ose_training.models.interp(b.input, b)
    tinterp = interp(b.input, b)
    merged = xr.merge([nad, duacs_nad.rename(ssh='duacs')])
    merged.assign(
        torch_interp = ('time', tinterp.numpy()[0]),
    ).reset_index('time', drop=True).isel(time=slice(300,1400)).to_array().plot(hue='variable', figsize=(10,5))

    merged.assign(
        torch_interp = ('time', tinterp.numpy()[0]),
    ).reset_index('time', drop=True).isel(time=900)


    tgt_c = contrib.ose_training.models.scale_tgt_coords(b)
    tgt_c[0,900]
    ((tgt_c[0,900] + 1) / 2 * torch.tensor([10,240, 240]))
    b.input[0][tuple(((tgt_c[0,900] + 1) / 2 * torch.tensor([10,240, 240])).round().int())]
    F.grid_sample(
        einops.rearrange(b.input[:, None, :,:,:], 'b c z y x -> b c x y z'),
        tgt_c[None, None, ..., 900:901, :],
        mode='nearest', align_corners=False
    )

    b.input_coords.lat
    import matplotlib.pyplot as plt
    duacs.dims
    plt.imshow(b.input.numpy()[0,0])

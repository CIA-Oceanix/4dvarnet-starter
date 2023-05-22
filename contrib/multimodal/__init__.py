import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import collections
import src.data
import src.models
import src.utils

MultiModalSSTTrainingItem = collections.namedtuple(
    "MultiModalSSTTrainingItem", ["input", "tgt", "sst"]
)


def load_data_with_sst(obs_var='five_nadirs'):
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

    sst = (
        xr.open_dataset(
            "../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_sst_y2013.1y.nc"
        )
        .sst.isel(time=slice(0, -1))
        .interp(lat=inp.lat, lon=inp.lon, method="nearest")
    )

    ds =  (
        xr.Dataset(dict(
            input=inp,
            tgt=(gt.dims, gt.values),
            sst=(sst.dims, sst.values)
        ), inp.coords).load()
        .transpose('time', 'lat', 'lon')
    )
    return ds.to_array()



class MultiModalDataModule(src.data.BaseDataModule):
    def post_fn(self):

        normalize_ssh = lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1]
        m_sst, s_sst = self.train_mean_std('sst')
        normalize_sst = lambda item: (item - m_sst) / s_sst
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                MultiModalSSTTrainingItem._make,
                lambda item: item._replace(tgt=normalize_ssh(item.tgt)),
                lambda item: item._replace(input=normalize_ssh(item.input)),
                lambda item: item._replace(sst=normalize_sst(item.sst)),
            ],
        )

class MultiModalObsCost(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.base_cost = src.models.BaseObsCost()

        self.conv_ssh =  torch.nn.Conv2d(dim_in, dim_hidden, (3, 3), padding=1, bias=False)
        self.conv_sst =  torch.nn.Conv2d(dim_in, dim_hidden, (3, 3), padding=1, bias=False)

    def forward(self, state, batch):
        ssh_cost =  self.base_cost(state, batch)
        sst_cost =  torch.nn.functional.mse_loss(
            self.conv_ssh(state),
            self.conv_sst(batch.sst.nan_to_num()),
        )
        return ssh_cost + sst_cost

class NonLinearMultiModalObsCost(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.base_cost = src.models.BaseObsCost()
        conv = lambda i, o: torch.nn.Conv2d(i, o, (3, 3), padding=1, bias=False)
        self.head = torch.nn.Sequential(
            torch.nn.ReLU(),
            conv(dim_hidden, dim_hidden),
            torch.nn.Tanh(),
            conv(dim_hidden, dim_hidden),
            torch.nn.BatchNorm2d(dim_hidden, track_running_stats=False)
        )
        self.leg_ssh = torch.nn.Sequential(
            conv(dim_in, 2*dim_hidden),
            torch.nn.ReLU(),
            conv(2*dim_hidden, dim_hidden),
        )

        self.leg_sst = torch.nn.Sequential(
            conv(dim_in, 2*dim_hidden),
            torch.nn.ReLU(),
            conv(2*dim_hidden, dim_hidden),
        ) 

    def forward(self, state, batch):
        ssh_cost =  self.base_cost(state, batch)
        sst_cost =  torch.nn.functional.mse_loss(
            self.head(self.leg_ssh(state)),
            self.head(self.leg_sst(batch.sst.nan_to_num())),
        )
        return ssh_cost + sst_cost


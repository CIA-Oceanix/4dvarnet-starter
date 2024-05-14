import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import collections
import src.data
import src.models
import src.utils
from copy import deepcopy
import torch
import xarray as xr
import kornia.filters as kfilts
from src.data import AugmentedDataset, BaseDataModule, XrDataset

MultiModalSSTTrainingItem = collections.namedtuple(
    "MultiModalSSTTrainingItem", ["input", "tgt", "sst"]
)

def threshold_xarray(da):
    threshold = 10**3
    da = xr.where(da > threshold, 0, da)
    da = xr.where(da <= 0, 0, da)
    return da

def load_natl_data_sst(tgt_path, tgt_var, inp_path, inp_var, sst_path, sst_var, **kwargs):
    tgt = (
        xr.open_dataset(tgt_path)[tgt_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        .pipe(threshold_xarray)
    )
    inp = (
        xr.open_dataset(inp_path)[inp_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        .pipe(threshold_xarray)
        #.pipe(mask)
    )
    sst = (
        xr.open_dataset(sst_path)[sst_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        #.pipe(mask)
    )
    print(xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array())
    return (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values), sst = (sst.dims, sst.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array()
    )


def load_data_with_sst(obs_var='five_nadirs'):
    inp = xr.open_dataset( "../sla-data-registry/CalData/cal_data_new_errs.nc")[obs_var]
    gt = ( xr.open_dataset(
            "../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc"
        ).ssh.isel(time=slice(0, -1))
        .interp(lat=inp.lat, lon=inp.lon, method="nearest")
    )
    sst = (xr.open_dataset(
            "../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_sst_y2013.1y.nc"
        ).sst.isel(time=slice(0, -1))
        .interp(lat=inp.lat, lon=inp.lon, method="nearest")
    )
    ds =  (
        xr.Dataset(dict(
            input=inp, tgt=(gt.dims, gt.values), sst=(sst.dims, sst.values)
        ), inp.coords).load()
        .transpose('time', 'lat', 'lon')
    )
    return ds.to_array()

class MultiModalDataModule(src.data.BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_std_domain = kwargs.get('mean_std_domain', 'train')
        self.std_c = kwargs.get('std_c', 1.)

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
    
    def norm_stats(self):
        if self._norm_stats is None:
            if self.norm_type == 'z_score':
                self._norm_stats = self.train_mean_std()
                print("Norm stats", self._norm_stats)
            if self.norm_type == 'min_max':
                self._norm_stats = self.min_max_norm()
                print("Norm stats", self._norm_stats)
        return self._norm_stats
    
    def train_mean_std(self, variable='tgt'):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains[self.mean_std_domain])
        return train_data.sel(variable=variable).pipe(lambda da: (da.mean().values.item(), self.std_c*da.std().values.item()))
    
    def min_max_norm(self, variable = 'tgt'):
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains[self.mean_std_domain])
        min_value = train_data.sel(variable=variable).min().values.item()
        max_value = train_data.sel(variable=variable).max().values.item()
        return min_value, max_value

    def setup(self, stage='test'):
        post_fn = self.post_fn()

        if stage == 'fit':
            train_data = self.input_da.sel(self.domains['train'])
            train_xrds_kw = deepcopy(self.xrds_kw)
            
            self.train_ds = XrDataset(
                train_data, **train_xrds_kw, postpro_fn=post_fn,
            )
            if self.aug_kw:
                self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = XrDataset(
                self.input_da.sel(self.domains['val']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )
        else:
            self.test_ds = XrDataset(
                self.input_da.sel(self.domains['test']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )
class TransfertDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_std_domain = kwargs.get('mean_std_domain', 'train')
        self.std_c = kwargs.get('std_c', 1.)

    

class MultiModalObsCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, ecs_weight, sst_weight, weight1 = 1.):
        super().__init__()
        self.base_cost = src.models.BaseObsCost()

        self.conv_ssh =  torch.nn.Conv2d(dim_in, dim_hidden, (3, 3), padding=1, bias=False)
        self.conv_sst =  torch.nn.Conv2d(dim_in, dim_hidden, (3, 3), padding=1, bias=False)
        
        self.weight1_torch    = torch.nn.Parameter(torch.tensor(weight1), requires_grad = True)
        self.ssh_weight_torch = torch.nn.Parameter(torch.tensor(ecs_weight), requires_grad = True)
        self.sst_weight_torch = torch.nn.Parameter(torch.tensor(sst_weight), requires_grad = True)

    def forward(self, state, batch):

        ssh_cost =  self.base_cost(state, batch)
        sst_cost =  torch.nn.functional.mse_loss(
            self.conv_ssh(state),
            self.conv_sst(batch.sst.nan_to_num()),
        )

        final_cost = self.ssh_weight_torch * ssh_cost + self.sst_weight_torch * sst_cost
    
        return final_cost
    
class Lit4dVarNet_SST(src.models.Lit4dVarNet):
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        # Create a mask selecting non-NaN values
        # if self.mask_sampling_with_nan is not None:
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
        
            weight_obs = self.solver.obs_cost.weight1_torch
            weight_prior = self.solver.prior_cost.weight3_torch
            self.log('sampling_rate', self.sampling_rate, on_step=False, on_epoch=True)
            self.log('weight obs', weight_obs , on_step=False, on_epoch=True)
            self.log('weight prior', weight_prior,on_step=False, on_epoch=True)

            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss
            #training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
            self.log( "ecs_weight", self.solver.obs_cost.ssh_weight_torch, prog_bar=True, on_step=False, on_epoch=True)
            self.log( "sst_weight", self.solver.obs_cost.sst_weight_torch, prog_bar=True, on_step=False, on_epoch=True)
            return training_loss, out
        
        else:
            loss, out = self.base_step(batch, phase)
            return loss, out

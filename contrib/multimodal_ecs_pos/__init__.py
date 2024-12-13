import xarray as xr
import einops
import functools as ft
import torch
import torch.nn as nn
import collections
import src.data
import src.models
import src.utils
import kornia.filters as kfilts

MultiModalPosSSTTrainingItem = collections.namedtuple(
    "MultiModalPosSSTTrainingItem", ["input", "tgt", "sst", "pos"]
)

def threshold_xarray(da):
    threshold = 10**3
    da = xr.where(da > threshold, 0, da)
    da = xr.where(da <= 0, 0, da)
    return da

def load_natl_data_sst_pos(tgt_path, tgt_var, inp_path, inp_var, sst_path, sst_var, pos_path, pos_var, **kwargs):
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
    pos = (
        xr.open_dataset(pos_path)[pos_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        #.pipe(mask)
    )
    return (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values), sst = (sst.dims, sst.values), pos = (pos.dims, pos.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array()
    )

class MultiModalPosSSTDataModule(src.data.BaseDataModule):
    def post_fn(self):
        normalize_ssh = lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1]
        m_sst, s_sst = self.train_mean_std('sst')
        normalize_sst = lambda item: (item - m_sst) / s_sst
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                MultiModalPosSSTTrainingItem._make,
                lambda item: item._replace(tgt=normalize_ssh(item.tgt)),
                lambda item: item._replace(input=normalize_ssh(item.input)),
                lambda item: item._replace(sst=normalize_sst(item.sst)),
                lambda item: item._replace(pos=(item.pos)),
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
        #is_nan_ssh_cost = torch.isnan(ssh_cost)
        final_cost = ssh_cost + sst_cost #if not is_nan_ssh_cost else sst_cost
        return final_cost

    
class Lit4dVarNetPosSST(src.models.Lit4dVarNet):
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None
        mask = (batch.pos == 0)
        masked_input = torch.where(mask, torch.tensor(float('nan')), batch.input)
        #masked_input = batch.input / batch.pos
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
        mask = (batch.pos == 0)
        masked_input = torch.where(mask, torch.tensor(float('nan')), batch.input)
        #masked_input = batch.input / batch.pos
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

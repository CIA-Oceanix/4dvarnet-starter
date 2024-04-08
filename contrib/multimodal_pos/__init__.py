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

MultiModalPosTrainingItem = collections.namedtuple(
    "MultiModalPosTrainingItem", ["input", "tgt", "pos"]
)

def threshold_xarray(da):
    threshold = 10**3
    da = xr.where(da > threshold, 0, da)
    da = xr.where(da <= 0, 0, da)
    return da

def load_natl_data_pos(tgt_path, tgt_var, inp_path, inp_var, pos_path, pos_var, **kwargs):
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
    pos = (
        xr.open_dataset(pos_path)[pos_var]
        .sel(kwargs.get('domain', None))
        .sel(kwargs.get('period', None))
        #.pipe(mask)
    )
    print(xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values), pos = (pos.dims, pos.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array())
    return (
        xr.Dataset(
            dict(input=inp, tgt=(tgt.dims, tgt.values), pos = (pos.dims, pos.values)),
            inp.coords,
        )
        .transpose('time', 'lat', 'lon')
        .to_array()
    )

class MultiModalPosDataModule(src.data.BaseDataModule):
    def post_fn(self):
        
        normalize_ecs = lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1]
        return ft.partial(
            ft.reduce,
            lambda i, f: f(i),
            [
                MultiModalPosTrainingItem._make,
                lambda item: item._replace(tgt=normalize_ecs(item.tgt)),
                lambda item: item._replace(input=normalize_ecs(item.input)),
                lambda item: item._replace(pos=(item.pos)),
            ],
        )
    
class Lit4dVarNetPos(src.models.Lit4dVarNet):
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

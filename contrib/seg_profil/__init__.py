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
import torch.nn.functional as F
    
class Lit4dVarNet_Seg(src.models.Lit4dVarNet):
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
            ecs_class_O = torch.where(batch.tgt!= 0, torch.ones_like(batch.tgt), torch.zeros_like(batch.tgt)).squeeze(dim=1).float()
            out_classif_O = torch.where(out!= 0, torch.ones_like(out), torch.zeros_like(out)).squeeze(dim=1).float()
            binary_cross_entropy_loss = - F.binary_cross_entropy_with_logits(ecs_class_O, out_classif_O)

            grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
            prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_cost", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_class_loss", binary_cross_entropy_loss, prog_bar=True, on_step=False, on_epoch=True)
            
            training_loss = 10 * loss + 20 * prior_cost + 5 * grad_loss + binary_cross_entropy_loss
            
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
                    # add the card
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
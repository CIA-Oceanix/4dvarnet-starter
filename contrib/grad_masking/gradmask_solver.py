import torch
import torch.nn.functional as F
import kornia.filters as kfilts

from src.models import GradSolverZero, BilinAEPriorCost, Lit4dVarNet, Lit4dVarNetForecast
from contrib.forecast_plus.models import Plus4dVarNetForecast

"""
    Overriden classes from src.models

    The idea is to provide a mask of continents to the Prior Solver, so that the Gradient of the Prior Cost does not contain 'useless' information of continents reconstruction
"""

class GradMaskLit4dVarNet(Lit4dVarNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)

        grad_mask = batch.tgt.isfinite()

        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out), grad_mask)
        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

class GradMaskLit4dVarNetForecast(Lit4dVarNetForecast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        
        grad_mask = batch.tgt.isfinite()

        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out), grad_mask)
        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out
    
class GradMaskPlus4dVarNetForecast(Plus4dVarNetForecast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.1:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse(kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        
        grad_mask = batch.tgt.isfinite()

        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out), grad_mask)
        self.log(f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

class GradMaskSolverZero(GradSolverZero):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def solver_step(self, state, batch, step):

        grad_mask = batch.tgt.isfinite()

        var_cost = self.prior_cost(state, grad_mask) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
            + self.lr_grad * (step + 1) / self.n_step * grad
        )

        return state - state_update
    
class GradMaskBilinAEPriorCost(BilinAEPriorCost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, state, mask):
        mask = mask
        return F.mse_loss(state[mask], self.forward_ae(state)[mask])
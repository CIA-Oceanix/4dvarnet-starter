import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradSolver_Lgv(nn.Module):
    def __init__(self, score_model, nlpobs, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__()
        self.score_model = score_model
        self.nlpobs = nlpobs
        self.grad_mod = grad_mod
        self.n_step = n_step
        self.time_steps = np.linspace(1., 1e-3, self.n_step)
        self.step_size = self.time_steps[0] - self.time_steps[1]
        self.lr_grad = lr_grad
        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init
        return batch.input.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, step,
                          use_noise=True,
                          use_conditioning=True):

        # define some parameters
        device = state.device
        batch_time_step = torch.ones(state.size()[0], device=device) * self.time_steps[step]

        # 1) Corrector step (Langevin MCMC)

        # compute Grad(log(p(x|y)))
        lpobs = -1.*torch.nanmean(self.nlpobs(state,batch,transpose=False))
        grad1 = self.score_model(state, batch_time_step) 
        grad2 = torch.autograd.grad(lpobs, state, create_graph=True)[0]
        if use_conditioning:
            grad = grad1 + grad2
        else:
            grad = grad1
        # add noise
        noise = torch.randn(grad.size(),requires_grad=True).to(device)
        if use_noise:
            gmod = self.grad_mod(grad) #+ np.sqrt(2)*noise)
        else:
            gmod = self.grad_mod(grad)
        state_update = 1 / (step + 1) * gmod
        state = state + state_update #+ noise

        # Predictor step (Euler-Maruyama)
        '''
        g = diffusion_coeff(batch_time_step)
        state_mean = state + (g**2)[:, None, None, None] * self.score_model(state, batch_time_step) * self.step_size
        state = state_mean + torch.sqrt(g**2 * self.step_size)[:, None, None, None] * torch.randn_like(state)
        '''

        return state
        
    def forward(self, batch, use_conditioning=True):

        device = batch.input.device

        with torch.set_grad_enabled(True):

            #state = self.init_state(batch)
            t = torch.ones(batch.input.size()[0], device=device)
            x_init = torch.randn(batch.input.size(),requires_grad=True).to(device) \
                     * self.score_model.marginal_prob_std(t)[:, None, None, None]
            state = self.init_state(batch, x_init = x_init)

            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step, use_conditioning = use_conditioning)
                if not self.training:
                    state = state.detach().requires_grad_(True)

        return state

class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        x =  x / self._grad_norm
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        return out

### import pandas as pd
import einops
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from contrib.stoch_spde.solver import GradSolver_Lgv
from src.models import GradSolver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Upsampler(torch.nn.Module):
    def __init__(self, scale_factor, mode, align_corners, antialias, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias
        
    def forward(self, x):        
        x = F.interpolate(x, scale_factor=self.scale_factor,
                          mode=self.mode, align_corners=self.align_corners,
                          antialias=self.antialias)
        return x
    
class GradSolver_wcov(GradSolver):

    def init_state(self, batch, x_init=None):
        x_init = super().init_state(batch, x_init)
        x_covs = torch.cat((batch.u, batch.v), dim=1)
        return (x_init, x_covs)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state[0], batch)
        x, x_covs = state
        grad = torch.autograd.grad(var_cost, x, create_graph=True)[0]
        x_update = (
            1 / (step + 1) * self.grad_mod(grad)
            + self.lr_grad * (step + 1) / self.n_step * grad
        )
        state = (x - x_update, x_covs)
        return state

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = [s.detach().requires_grad_(True) for s in state]

            if not self.training:
                state = [self.prior_cost.forward_ae(state), state[1]]

        return state[0]

class GradSolver_spde_wcov(GradSolver_Lgv):

    def init_state(self, batch, x_init=None):
        x_init = super().init_state(batch, x_init)
        x_covs = torch.cat((batch.u, batch.v), dim=1)
        return (x_init, x_covs)

    def solver_step(self, state, batch, step, mu=None, noise=False):
        device = state[0].device
        n_b, n_t, n_y, n_x = batch.input.shape
        if self.aug_state==False:
            x = state[0]
            theta = None
            x_covs = state[1]
        else:
            x = state[0][:,:n_t,:,:]
            theta = state[0][:,n_t:,:,:]
            x_covs = state[1]
        if self.aug_state==False:
            grad_logp = self.lambda_obs**2 * self.obs_cost(x, batch) +\
                        self.lambda_reg**2 * self.prior_cost(state,exclude_params=self.exclude_params) 
        else:
            if not self.unet_prior:
                grad_logp = torch.nanmean(self.lambda_obs**2 * self.obs_cost(x,batch) + \
                                          self.lambda_reg**2 * (-1) * self.nll(x, theta, mu=mu, det=False))
            else: 
                grad_logp = self.lambda_obs**2 * self.obs_cost(x, batch) +\
                            self.lambda_reg**2 * self.prior_cost(state,exclude_params=self.exclude_params) 
        grad = torch.autograd.grad(grad_logp, state[0], create_graph=True)[0]

        # B,t,x,y -> b,t,y,x
        # add noise
        if noise:
            noise = torch.randn(grad.size(),requires_grad=True).to(device) #* self.lr_grad
            gmod = self.grad_mod(grad + noise)
        else:
            gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        state_update = 1 / (step + 1) * gmod

        res = state[0] - state_update #+ noise

        # constrain theta
        if self.aug_state==True:
            idx_kappa = torch.arange(n_t,2*n_t)
            idx_tau = torch.arange(2*n_t,3*n_t)
            idx_m1 = torch.arange(3*n_t,4*n_t)
            idx_m2 = torch.arange(4*n_t,5*n_t)
            idx_vx = torch.arange(5*n_t,6*n_t)
            idx_vy = torch.arange(6*n_t,7*n_t)
            idx_gamma = torch.arange(7*n_t,8*n_t)
            idx_beta = torch.arange(8*n_t,9*n_t)
            
            kappa = res[:,idx_kappa,:,:]
            tau = res[:,idx_tau,:,:]
            m1 = res[:,idx_m1,:,:]
            m2 = res[:,idx_m2,:,:]
            vx = res[:,idx_vx,:,:]
            vy = res[:,idx_vy,:,:]
            gamma = res[:,idx_gamma,:,:]
            beta = res[:,idx_beta,:,:]
            
            # parameters: gamma > 0
            gamma =  F.relu(gamma)+.01
            res = torch.index_add(res,1,idx_gamma.to(device),-1*res[:,idx_gamma,:,:])
            res = torch.index_add(res,1,idx_gamma.to(device),gamma)

            # parameters: beta > 0
            beta =  F.relu(beta)+.01  
            res = torch.index_add(res,1,idx_beta.to(device),-1*res[:,idx_beta,:,:])
            res = torch.index_add(res,1,idx_beta.to(device),beta)             
            
            # parameters: kappa > 0
            kappa = F.relu(kappa)+.01
            res = torch.index_add(res,1,idx_kappa.to(device),-1*res[:,idx_kappa,:,:])
            res = torch.index_add(res,1,idx_kappa.to(device),kappa)

            # parameters: tau > 0
            tau = F.relu(tau)+.1
            res = torch.index_add(res,1,idx_tau.to(device),-1*res[:,idx_tau,:,:])
            res = torch.index_add(res,1,idx_tau.to(device),tau)
            
        return (res, x_covs)
        
    def forward(self, batch, x_init = None, mu=None, reshape_theta=True):

        device = batch.input.device
        n_t = batch.input.size(1)
        
        with torch.set_grad_enabled(True):
            state = self.init_state(batch, x_init=x_init)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, mu=mu, step=step)
                if not self.training:
                    state = [s.detach().requires_grad_(True) for s in state]

        if self.aug_state==False:
            theta = None                                            
        else:
            theta = state[0][:,n_t:,:,:]
            state = state[0][:,:n_t,:,:]
            if reshape_theta:
                kappa, m, H, tau = self.nll.reshape_params(theta)
                theta = [kappa, m, H, tau]
        
        return state, theta

class BilinAEPriorCost_wcov(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True, ncov=2):
        super().__init__()
        self.ncov = ncov
        self.bilin_quad = bilin_quad
        self.conv_in = nn.Conv2d(
            dim_in*(self.ncov+1), dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.bilin_1 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.bilin_21 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.down2 = nn.MaxPool2d(downsamp) if downsamp is not None else nn.Identity()

        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward_ae(self, state):
        x = self.down(state[0])
        x_covs = self.down2(state[1])
        x = self.conv_in(torch.cat((x,x_covs),dim=1))
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state):
        return F.mse_loss(state[0], self.forward_ae(state))

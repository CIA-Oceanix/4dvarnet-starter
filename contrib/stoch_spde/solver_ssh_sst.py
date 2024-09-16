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
#import matplotlib.pyplot as plt

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
    
class GradSolver_Lgv(nn.Module):
    def __init__(self, nll, nlpobs, obs_cost, grad_mod, n_step, prior_cost=None,
                 lr_grad=0.2, exclude_params=False, aug_state=True,  unet_prior=False, **kwargs):
        super().__init__()
        self.nll = nll
        self.nlpobs = nlpobs
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.unet_prior = unet_prior
        self.grad_mod = grad_mod
        self.n_step = n_step
        self.lr_grad = lr_grad
        self.exclude_params = exclude_params
        self._grad_norm = None
        self.aug_state = aug_state
        self.lambda_obs = torch.nn.Parameter(torch.Tensor([1.]))
        self.lambda_reg = torch.nn.Parameter(torch.Tensor([1.]))

        self.downsamp = 2#self.nll.downsamp
        self.down = nn.AvgPool2d(self.downsamp) if self.downsamp is not None else nn.Identity()
        
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=self.downsamp)
            if self.downsamp is not None
            else nn.Identity()
        )
    def custom_sigmoid(self,x,min,max):
        add_const = min/(max-min)
        mult_const = max-min
        return (torch.sigmoid(x)+add_const)*mult_const

    def interpolate_na_2D(self, da, max_value=100.):
        return (
            da.where(np.abs(da) < max_value, np.nan)
            .pipe(lambda da: da)
            .to_dataframe()
            .interpolate(limit_direction='both')
            .pipe(xr.Dataset.from_dataframe)
        )

    def init_state(self, batch, x_init=None):
        n_b, n_t, n_y, n_x = batch.input.shape
        n_t = n_t//2
        if self.aug_state==False:
            if x_init is not None:
                return x_init
            else:
                return batch.input.nan_to_num().detach().requires_grad_(True)        
        else:
            if x_init is None:
                x_init = batch.input.nan_to_num().detach()
            x_init_ssh = x_init[:,:n_t,:,:]
            x_init_sst = x_init[:,n_t:(2*n_t),:,:]
            kappa_ssh_init = torch.sum(torch.abs(kfilts.spatial_gradient(x_init_ssh,normalized=True)),dim=2)
            kappa_ssh_init = torch.max(kappa_ssh_init)-kappa_ssh_init+.01
            tau_ssh_init = torch.sum(torch.abs(kfilts.spatial_gradient(x_init_ssh,normalized=True)),dim=2)+.01
            m1_ssh_init = kfilts.spatial_gradient(x_init_ssh,normalized=True)[:,:,0,:,:]
            m2_ssh_init = kfilts.spatial_gradient(x_init_ssh,normalized=True)[:,:,1,:,:]
            vx_ssh_init = kfilts.spatial_gradient(x_init_ssh,order=1,normalized=True)[:,:,0,:,:]
            vy_ssh_init = kfilts.spatial_gradient(x_init_ssh,order=1,normalized=True)[:,:,1,:,:]
            gamma_ssh_init = torch.ones(x_init_ssh.size()).to(device)
            beta_ssh_init = torch.ones(x_init_ssh.size()).to(device)
            kappa_sst_init = torch.sum(torch.abs(kfilts.spatial_gradient(x_init_sst,normalized=True)),dim=2)
            kappa_sst_init = torch.max(kappa_sst_init)-kappa_sst_init+.01
            tau_sst_init = torch.sum(torch.abs(kfilts.spatial_gradient(x_init_sst,normalized=True)),dim=2)+.01
            m1_sst_init = kfilts.spatial_gradient(x_init_sst,normalized=True)[:,:,0,:,:]
            m2_sst_init = kfilts.spatial_gradient(x_init_sst,normalized=True)[:,:,1,:,:]
            vx_sst_init = kfilts.spatial_gradient(x_init_sst,order=1,normalized=True)[:,:,0,:,:]
            vy_sst_init = kfilts.spatial_gradient(x_init_sst,order=1,normalized=True)[:,:,1,:,:]
            gamma_sst_init = torch.ones(x_init_sst.size()).to(device)
            beta_sst_init = torch.ones(x_init_sst.size()).to(device)
            state_init =  torch.cat((x_init,
                                kappa_ssh_init,
                                tau_ssh_init,
                                m1_ssh_init,
                                m2_ssh_init,
                                vx_ssh_init,
                                vy_ssh_init,
                                gamma_ssh_init,
                                beta_ssh_init,
                                kappa_sst_init,
                                tau_sst_init,
                                m1_sst_init,
                                m2_sst_init,
                                vx_sst_init,
                                vy_sst_init,
                                gamma_sst_init,
                                beta_sst_init),dim=1).requires_grad_(True)
            return state_init

    def solver_step(self, state, batch, step, mu=None, noise=False):
        device = state.device
        n_b, n_t, n_y, n_x = batch.input.shape
        n_t = n_t//2
        if self.aug_state==False:
            x = state
            theta = None
        else:
            x = state[:,:(2*n_t),:,:]
            theta = state[:,(2*n_t):,:,:]
            x_ssh = x[:,:n_t,:,:]
            x_sst = x[:,n_t:(2*n_t),:,:]
            theta_ssh = theta[:,:(8*n_t),:,:]
            theta_sst = theta[:,(8*n_t):,:,:]
        if self.aug_state==False:
            grad_logp = self.lambda_obs**2 * self.obs_cost(x, batch) +\
                        self.lambda_reg**2 * self.prior_cost(state,exclude_params=self.exclude_params) 
        else:
            if not self.unet_prior:
                if mu is not None:
                    mu_ssh, mu_sst = mu[:,:n_t,:,:], mu[:,n_t:,:,:]
                else:
                    mu_ssh = None
                    mu_sst = None

                grad_logp = torch.nanmean(self.lambda_obs**2 * self.obs_cost(x,batch) + \
                                          self.lambda_reg**2 * (-1) * ( self.nll(x_ssh, theta_ssh, mu=mu_ssh, det=False) + \
                                                                        self.nll(x_sst, theta_sst, mu=mu_sst, det=False) ) ) 
            else: 
                grad_logp = self.lambda_obs**2 * self.obs_cost(x, batch) +\
                            self.lambda_reg**2 * self.prior_cost(state,exclude_params=self.exclude_params) 
        grad = torch.autograd.grad(grad_logp, state, create_graph=True)[0]

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

        res = state - state_update #+ noise

        def constrain_params(res, i_start):
            idx_kappa = torch.arange(i_start*n_t,(i_start+1)*n_t)
            idx_tau = torch.arange((i_start+1)*n_t,(i_start+2)*n_t)
            idx_m1 = torch.arange((i_start+2)*n_t,(i_start+3)*n_t)
            idx_m2 = torch.arange((i_start+3)*n_t,(i_start+4)*n_t)
            idx_vx = torch.arange((i_start+4)*n_t,(i_start+5)*n_t)
            idx_vy = torch.arange((i_start+5)*n_t,(i_start+6)*n_t)
            idx_gamma = torch.arange((i_start+6)*n_t,(i_start+7)*n_t)
            idx_beta = torch.arange((i_start+7)*n_t,(i_start+8)*n_t)
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
            
            return res

        # constrain theta
        if self.aug_state==True:
            # constrain theta_ssh
            res = constrain_params(res, i_start=2)
            # constrain theta_sst
            res = constrain_params(res, i_start=10)

        return res

    def forward(self, batch, x_init = None, mu=None, reshape_theta=True):

        device = batch.input.device
        n_t = batch.input.size(1)
        n_t = n_t//2
        
        with torch.set_grad_enabled(True):
            state = self.init_state(batch, x_init=x_init)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, mu=mu, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

        if self.aug_state==False:
            theta = None                                            
        else:
            '''
            # plot
            i_start = 2
            data = xr.Dataset(data_vars=dict(
                tau_ssh=(["time", "lat", "lon"], state[0,((i_start+1)*n_t):((i_start+2)*n_t),:,:].detach().cpu().data),
                               ),
                          coords=(dict(time=range(5),
                                  lon=range(120),
                                  lat=range(120))))
            data.tau_ssh.plot(col='time',col_wrap=5)
            plt.show()
            '''

            theta_ssh = state[:,(2*n_t):(10*n_t),:,:]
            theta_sst = state[:,(10*n_t):,:,:]
            state = state[:,:2*n_t,:,:]
            if reshape_theta:
                kappa_ssh, m_ssh, H_ssh, tau_ssh = self.nll.reshape_params(theta_ssh)
                theta_ssh = [kappa_ssh, m_ssh, H_ssh, tau_ssh]
                kappa_sst, m_sst, H_sst, tau_sst = self.nll.reshape_params(theta_sst)
                theta_sst = [kappa_sst, m_sst, H_sst, tau_sst]
            theta = theta_ssh + theta_sst
        return state, theta

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

        self.dropout = torch.nn.Dropout(dropout)
        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

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

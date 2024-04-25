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
            #Upsampler(scale_factor=self.downsamp,
            #         mode='bilinear',
            #         align_corners=False,
            #         antialias=True)
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

    def init_fields(self, obs):
        n_b, n_t, n_y, n_x = obs.shape
        y_xr = xr.Dataset(data_vars={'ssh':(('batch','time','lat','lon'),obs.detach().cpu())},
                      coords={'batch':np.arange(n_b),
                              'time':np.arange(n_t),
                              'lon':np.arange(n_x),
                              'lat':np.arange(n_y)})
        ybis =  self.interpolate_na_2D(y_xr.mean(dim='time'))
        ybis =  ybis.coarsen(lon=4, boundary="trim").mean().coarsen(lat=4, boundary="trim").mean()
        ybis = einops.repeat(torch.tensor(ybis.ssh.data), 'b m n -> b k m n', k=5)
        m = torch.nn.Upsample(scale_factor=4, mode='bilinear')  # align_corners=False
        ybis = m(ybis)
        qg_veloc = kfilts.spatial_gradient(ybis,normalized=True)
        qg_veloc = torch.permute(qg_veloc,(0,2,1,3,4))
        return qg_veloc
    
    def init_state(self, batch, x_init=None):
        
        if self.aug_state==False:
            if x_init is not None:
                return x_init
            else:
                return batch.input.nan_to_num().detach().requires_grad_(True)        
        else:
            #qg_veloc = self.init_fields(batch.input.detach()).to(device)
            #m1_init = qg_veloc[:,0,:,:,:]
            #m2_init = qg_veloc[:,1,:,:,:]
            kappa_init = torch.sum(torch.abs(kfilts.spatial_gradient(x_init,normalized=True)),dim=2)
            kappa_init = torch.max(kappa_init)-kappa_init+.01
            tau_init = torch.sum(torch.abs(kfilts.spatial_gradient(x_init,normalized=True)),dim=2)+.01
            m1_init = kfilts.spatial_gradient(x_init,normalized=True)[:,:,0,:,:]
            m2_init = kfilts.spatial_gradient(x_init,normalized=True)[:,:,1,:,:]
            vx_init = kfilts.spatial_gradient(x_init,order=1,normalized=True)[:,:,0,:,:]
            vy_init = kfilts.spatial_gradient(x_init,order=1,normalized=True)[:,:,1,:,:]
            gamma_init = torch.ones(batch.input.size()).to(device)
            beta_init = torch.ones(batch.input.size()).to(device)
            '''
            kappa_init = torch.ones(batch.input.size()).to(device)*0.1
            tau_init = torch.ones(batch.input.size()).to(device)*1
            m1_init = torch.ones(batch.input.size()).to(device)*0.1
            m2_init = torch.ones(batch.input.size()).to(device)*0.1
            vx_init = torch.zeros(batch.input.size()).to(device)
            vy_init = torch.zeros(batch.input.size()).to(device)
            gamma_init = torch.ones(batch.input.size()).to(device)
            beta_init = torch.ones(batch.input.size()).to(device)
            '''
            """       
            H = []
            for k in range(n_t):
                vx_ = torch.reshape(vx_init[:,k,:,:],(n_b,n_x*n_y))
                vy_ = torch.reshape(vy_init[:,k,:,:],(n_b,n_x*n_y))
                vxy = torch.stack([vx_,vy_],dim=2)
                vxyT = torch.permute(vxy,(0,2,1))
                gamma_ = torch.reshape(gamma_init[:,k,:,:],(n_b,n_x*n_y))
                beta_ = torch.reshape(beta_init[:,k,:,:],(n_b,n_x*n_y))
                H_ = torch.einsum('ij,bk->bijk',
                              torch.eye(2).to(device),
                              gamma_)+\
                 torch.einsum('bk,bijk->bijk',beta_,torch.einsum('bki,bjk->bijk',vxy,vxyT))
                H.append(H_)
            H = torch.stack(H,dim=4)

            xr.Dataset(data_vars={'H11':(('time','lat','lon'),np.reshape(H[0,0,0,:,:9].t().detach().cpu(),(9,120,120)))},
                   coords={'time':np.arange(9),
                   'lon':np.arange(-66, -54, 0.1),
                   'lat':np.arange(32, 44, 0.1)}).H11.plot(col='time')
            plt.show()     
            xr.Dataset(data_vars={'H12':(('time','lat','lon'),np.reshape(H[0,0,1,:,:9].t().detach().cpu(),(9,120,120)))},
                   coords={'time':np.arange(9),
                   'lon':np.arange(-66, -54, 0.1),
                   'lat':np.arange(32, 44, 0.1)}).H12.plot(col='time')
            plt.show()   
            xr.Dataset(data_vars={'H21':(('time','lat','lon'),np.reshape(H[0,1,0,:,:9].t().detach().cpu(),(9,120,120)))},
                   coords={'time':np.arange(9),
                   'lon':np.arange(-66, -54, 0.1),
                   'lat':np.arange(32, 44, 0.1)}).H21.plot(col='time')
            plt.show()   
            xr.Dataset(data_vars={'H22':(('time','lat','lon'),np.reshape(H[0,1,1,:,:9].t().detach().cpu(),(9,120,120)))},
                   coords={'time':np.arange(9),
                   'lon':np.arange(-66, -54, 0.1),
                   'lat':np.arange(32, 44, 0.1)}).H22.plot(col='time')
            plt.show()   
            """ 
            if x_init is None:
                x_init = batch.input.nan_to_num().detach()
            state_init =  torch.cat((x_init,
                                kappa_init,
                                tau_init,
                                m1_init,
                                m2_init,
                                vx_init,
                                vy_init,
                                gamma_init,
                                beta_init),dim=1).requires_grad_(True)
            return state_init

    def solver_step(self, state, batch, step, mu=None, noise=False):
        device = state.device
        n_b, n_t, n_y, n_x = batch.input.shape
        #self.cnn = torch.nn.Conv2d(n_t,n_t,(3,3),padding=1,bias=False).to(device)
        #self.cnn2 = torch.nn.Conv2d(n_t,n_t,(3,3),padding=1,bias=False).to(device)
        #self.cnn3 = torch.nn.Conv2d(n_t,n_t,(3,3),padding=1,bias=False).to(device)
        #self.cnn4 = torch.nn.Conv2d(n_t,n_t,(3,3),padding=1,bias=False).to(device)
        if self.aug_state==False:
            x = state
            theta = None
        else:
            x = state[:,:n_t,:,:]
            theta = state[:,n_t:,:,:]
        if self.aug_state==False:
            grad_logp = self.lambda_obs**2 * self.obs_cost(x, batch) +\
                        self.lambda_reg**2 * self.prior_cost(state,exclude_params=self.exclude_params) 
        else:
            if not self.unet_prior:
                #grad_logp = torch.nanmean(self.lambda_obs**2 * self.nlpobs(x,batch) + \
                #                          self.lambda_reg**2 * (-1) * self.nll(x, theta, mu=mu, det=False))
                grad_logp = torch.nanmean(self.lambda_obs**2 * self.obs_cost(x,batch) + \
                                          self.lambda_reg**2 * (-1) * self.nll(x, theta, mu=mu, det=False))
            else: 
                grad_logp = self.lambda_obs**2 * self.obs_cost(x, batch) +\
                            self.lambda_reg**2 * self.prior_cost(state,exclude_params=self.exclude_params) 
        #m1_cost = F.mse_loss(self.cnn(theta[:,(2*n_t):(3*n_t),:,:]),self.cnn2(qg_veloc_x))
        #m2_cost = F.mse_loss(self.cnn3(theta[:,(3*n_t):(4*n_t),:,:]),self.cnn4(qg_veloc_y))
        #m1_cost = F.mse_loss(theta[:,(2*n_t):(3*n_t),:,:],qg_veloc_x)
        #m2_cost = F.mse_loss(theta[:,(3*n_t):(4*n_t),:,:],qg_veloc_y)
        #grad_logp += m1_cost + m2_cost
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
            
            '''
            H = []
            for k in range(n_t):
                vx_ = torch.reshape(vx[:,k,:,:],(n_b,n_x*n_y))
                vy_ = torch.reshape(vy[:,k,:,:],(n_b,n_x*n_y))
                vxy = torch.stack([vx_,vy_],dim=2)
                vxyT = torch.permute(vxy,(0,2,1))
                gamma_ = torch.reshape(gamma[:,k,:,:],(n_b,n_x*n_y))
                beta_ = torch.reshape(beta[:,k,:,:],(n_b,n_x*n_y))
                H_ = torch.einsum('ij,bk->bijk',
                              torch.eye(2).to(device),
                              gamma_)+\
                     torch.einsum('bk,bijk->bijk',beta_,torch.einsum('bki,bjk->bijk',vxy,vxyT))
                H.append(H_)
            H = torch.stack(H,dim=4) 
            H = torch.reshape(H,(n_b,2,2,n_x,n_y,n_t))
            H = torch.permute(H,(0,1,2,5,3,4))
            h11 = H[:,0,0,:,:,:]
            h22 = H[:,1,1,:,:,:]
            h12 = H[:,1,0,:,:,:]
            # check for Peclet condition           
            # parameters: m1 
            m1 = torch.where(m1/(2*tau*h11)<0.95,m1,torch.sign(m1)*0.9*2*tau*h11).to(device) # F.relu(gamma)+.01
            res = torch.index_add(res,1,torch.arange(3*n_t,4*n_t).to(device),-1*res[:,(3*n_t):(4*n_t),:,:])
            res = torch.index_add(res,1,torch.arange(3*n_t,4*n_t).to(device),m1)   
            # parameters: m2
            m2 = torch.where(torch.abs(m2)/(2*tau*h22)<0.95,m2,torch.sign(m2)*0.9*2*tau*h22).to(device) # F.relu(gamma)+.01
            res = torch.index_add(res,1,torch.arange(4*n_t,5*n_t).to(device),-1*res[:,(4*n_t):(5*n_t),:,:])
            res = torch.index_add(res,1,torch.arange(4*n_t,5*n_t).to(device),m2)           
            '''
            
        return res
        
    def forward(self, batch, x_init = None, mu=None, reshape_theta=True):

        device = batch.input.device
        n_t = batch.input.size(1)
        
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
            theta = state[:,n_t:,:,:]
            state = state[:,:n_t,:,:]
            if reshape_theta:
                kappa, m, H, tau = self.nll.reshape_params(theta)
                theta = [kappa, m, H, tau]
        
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

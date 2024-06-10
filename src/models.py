import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
import time
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image  # Import Pillow for GIF creation
import os


from src.QG_code.qgm import QGFV

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)

    @staticmethod
    def weighted_mse(err, weight):
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def forward(self, batch):
        return self.solver(batch)
    
    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        loss, out = self.base_step(batch, phase)
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, out))
        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)

        training_loss = 50 * loss + 1000 * grad_loss + 1.0 * prior_cost
        return training_loss, out

    def base_step(self, batch, phase=""):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        out = self(batch=batch)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                batch.input.cpu() * s + m,
                batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'out']

    def on_test_epoch_end(self):
        rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
            self.test_data, self.rec_weight.cpu().numpy()
        )

        if isinstance(rec_da, list):
            rec_da = rec_da[0]

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')
            self.logger.log_metrics(metrics.to_dict())


class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.input.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        

        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_ae(state)
        return state


class GradSolver_QG(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        # return batch.input.nan_to_num().detach().requires_grad_(True)
        return torch.zeros_like(batch.input).detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        

        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            #if not self.training:
                #state = self.prior_cost.forward_QG(state)
        return state


class GradSolver_Trained_Bilin_Phi(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.input.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        state_update = (
            1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
        

        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)

            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_ae(state)
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


class BaseObsCost(nn.Module):
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w=w

    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return self.w * F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True):
        super().__init__()
        self.bilin_quad = bilin_quad
        self.conv_in = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
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
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward_ae(self, x):
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state)) 


class Trained_BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True,path_to_checkpoint='/homes/g24meda/lab/4dvarnet-starter/outputs/2024-05-16/10-48-32/base_4_nadirs_DC_2020a_ssh/checkpoints/val_mse=5.00053-epoch=094.ckpt'):
        super().__init__()
        self.trained_prior_cost = BilinAEPriorCost(dim_in, dim_hidden, kernel_size, downsamp, bilin_quad)

        # Load the checkpoint
        checkpoint = torch.load(path_to_checkpoint)
        state_dict = checkpoint['state_dict']

        # Filter keys related to prior_cost
        prior_cost_state_dict = {k[len('solver.prior_cost.'):]: v for k, v in state_dict.items() if k.startswith('solver.prior_cost.')}

        # Load the state_dict into prior_cost
        self.trained_prior_cost.load_state_dict(prior_cost_state_dict)
        self.trained_prior_cost.eval()  # Set the model to evaluation mode

    def forward_ae(self, x):
        return self.trained_prior_cost.forward_ae(x)
    
    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state)) 

class LinearCost(nn.Module):
    def __init__(self, a=1, b=0) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward_linear(self, x):
        return self.a * x + self.b
    
    def forward(self, state):
        return F.mse_loss(state, self.forward_linear(state)) 
    
class QGCost(nn.Module):
    def __init__(self, nl, L, g_prime, H, f0, beta, tau0, bottom_drag_coef, apply_mask = False) -> None:
        super().__init__()
        self.nl = nl # number of layers of QG model, default = 1
        self.L = L # depth of the layer, default = 100 km ie 100000 m.
        self.g_prime = g_prime
        self.H = H
        self.f0 = f0
        self.beta = beta 
        self.tau0 = tau0
        self.bottom_drag_coef = bottom_drag_coef
        self.apply_mask = apply_mask

    def forward_QG(self, x):
        # take as input (in the case of batch size ==1) a tensor x of size [B=1,D,H,W] with B the size of batch, D the lenght of data assimilation window (typically 15 days), H and W the spatial size
        # return a vector of the same size with x[:,j+1,:,:] = QG(x[:,j,:,:])

        ## example for a single prediction x[:,j+1,:,:] = QG(x[:,j,:,:]), will need to make a loop later
        #ssh_0 = x[0][0].T
        #ssh_1 = self.forecast_QG(ssh_0).T     
        # return ssh_1  

        # x is of shape [B=1, D=15, H, W]
        print('new_forward')
        B, D, H, W = x.shape

        if B != 1:
            raise Exception("4dvarnet-QG does not handle batch with size > 1. Please set batch size to 1.") 

        # Initialize a tensor to store the forecasts 
        forecasts = torch.zeros_like(x)

        # Iterate over days of the batch
        for i in range(2-1): #range(D-1):
            # Extract the ssh of i-th day
            ssh_0 = x[0][i].T  # shape [H, W]

            # Perform the QG forecast of the next day
            ssh_1 = self.forecast_QG(ssh_0).T  

            # Store the forecast
            forecasts[0][i+1] = ssh_1

        # We can not compute QG forecast for the first day
        forecasts[0][0] = x[0][0]

        return forecasts[0][:2]

    
    def forward(self, state):
        #return F.mse_loss(state[0][0], self.forward_QG(state)) 
        return F.mse_loss(state[0][:2], self.forward_QG(state)) 
    
    def forecast_QG(self, ssh_0):
        ## provides the QG forecast of SSH 2D field for day N+1 given SSH field at day N .
        with torch.no_grad():
            torch.backends.cudnn.deterministic = True
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dtype = torch.float64

            # fill nan with 0
            #ssh_0 = torch.nan_to_num(ssh_0, nan=0.0)

            # grid
            nx = ssh_0.size()[0] - 1
            ny = ssh_0.size()[1] - 1
            dx = self.L / nx
            dy = self.L / ny
            xv = torch.linspace(-self.L/2, self.L/2, nx+1, dtype=torch.float64, device=device)
            yv = torch.linspace(-self.L/2, self.L/2, ny+1, dtype=torch.float64, device=device)
            x, y = torch.meshgrid(xv, yv, indexing='ij')

            H = torch.zeros(self.nl,1,1, dtype=dtype, device=device)
            if self.nl == 1:
                H[0,0,0] = self.H

            # gravity
            g_prime = torch.zeros(self.nl,1,1, dtype=dtype, device=device)
            if self.nl == 1:
                g_prime[0,0,0] = self.g_prime

            # Coriolis beta plane
            f = self.f0 + self.beta * (y - self.L/2)

            apply_mask = self.apply_mask

            # create rankine vortex in PV
            xc = 0.5 * (xv[1:] + xv[:-1])
            yc = 0.5 * (yv[1:] + yv[:-1])
            x, y = torch.meshgrid(xc, yc, indexing='ij')
            r = torch.sqrt(x**2 + y**2)
            # circular domain mask
            mask = (r < self.L/2).type(torch.float64) if apply_mask else torch.ones_like(x)

            param = {
                'nx': nx,
                'ny': ny,
                'nl': self.nl,
                'mask': mask,
                'n_ens': 1,
                'Lx': self.L,
                'Ly': self.L,
                'flux_stencil': 5,
                'H': H,
                'g_prime': g_prime,
                'tau0': self.tau0,
                'f0': self.f0,
                'beta': self.beta,
                'bottom_drag_coef': self.bottom_drag_coef,
                'device': device,
                'dt': 0, # time-step (s)
            }
            qg = QGFV(param)

            ## initialize psi_0 from ssh map
            g = 10.0 # m/s^2 
            psi_2d = (g/self.f0) * ssh_0
            qg.psi = psi_2d.unsqueeze(0).unsqueeze(0)

            ## initialize q_0 from psi_0
            qg.compute_q_from_psi()

            # compute u_max for CFL
            u, v = qg.grad_perp(qg.psi, qg.dx, qg.dy)
            u_norm_max = max(torch.abs(u).max().item(), torch.abs(v).max().item())
            
            ## rescaling factor 
            factor = torch.tensor(1/4, dtype=dtype, device=device) #(Ro * f0 * r0) / u_norm_max
            qg.psi *= factor
            qg.q *= factor
            u, v = qg.grad_perp(qg.psi, qg.dx, qg.dy)
            ##

            u_max = u.max().cpu().item()
            v_max = v.max().cpu().item()

            if u_max==0.0 or v_max==0.0:
                u_max = 4.0
                v_max = 4.0
            print(f'u_max {u_max:.2e}, v_max {v_max:.2e}')

            # set time step with CFL
            cfl = 1
            dt = cfl * min(dx / u_max, dy / v_max)
            qg.dt = dt
            print(f'integration time step : {round(dt)//3600} hour(s), {(round(dt)% 3600) // 60} minute(s), and {(round(dt)% 3600) % 60} second(s). ')

            # time params
            t = 0
            t_end = round(1 * 86400) # 1 day expressed in sec

            ##
            n_steps = int(t_end / dt) + 1 


            # plot, log, check nans
            freq_plot = int(t_end / 25 / dt) + 1
            freq_checknan = 10
            freq_log = int(t_end / 50 / dt) + 1

            t0 = time.time()
            for n in range(1, n_steps+1): #n_steps+1): #100
                qg.step()
                t += dt

            ssh_forecast = qg.psi * (self.f0/g) * (1/factor)
            ssh_forecast = ssh_forecast[0][0] # pass from [1,1,240,240] (one layer QG) -> [240,240]
            return ssh_forecast



    
# class QGCost_light(nn.Module):
#     def __init__(self, nl=1, L=100000, nx=None, ny=None) -> None:
#         super().__init__()
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.dtype = torch.float64

#         # QG parameters
#         self.nl = nl  # number of layers of QG model, default = 1
#         self.L = L  # depth of the layer, default = 100 km ie 100000 m.
#         self.nx = nx -1 # size of input vector
#         self.ny = ny -1 # size of input vector
#         self.g = 10.0 # m/s^2 , gravity constant
#         self.cfl = 0.5 # self.cfl
#         self.Bu = 1 # Burger number
#         self.Ro = 0.01 # Rosby number
    
#         # to be precomputed
#         self.dx = None
#         self.dy = None
#         self.f0 = None
#         self.r0 = None
#         self.qg = None
        
#         # Precompute static values
#         self.precompute_static_values()

#     def precompute_static_values(self):
#         with torch.no_grad():
#             torch.backends.cudnn.deterministic = True
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#             dtype = torch.float64

#             # QG parameters
#             nl = self.nl
#             L = self.L

#             # Grid
#             self.dx = L / self.nx
#             self.dy = L / self.ny
#             xv = torch.linspace(-L/2, L/2, self.nx+1, dtype=torch.float64, device=device)
#             yv = torch.linspace(-L/2, L/2, self.ny+1, dtype=torch.float64, device=device)
#             x, y = torch.meshgrid(xv, yv, indexing='ij')

#             H = torch.zeros(nl,1,1, dtype=dtype, device=device)
#             if nl == 1:
#                 H[0,0,0] = 1000.

#             # gravity
#             g_prime = torch.zeros(nl,1,1, dtype=dtype, device=device)
#             if nl == 1:
#                 g_prime[0,0,0] = 10

#             ## create rankine vortex
#             # Burger and Rossby numbers, coriolis set with Bu Number
#             self.r0, r1, r2 = 0.1*L, 0.1*L, 0.14*L
#             self.f0 = torch.sqrt(g_prime[0,0,0] * H[0,0,0] / self.Bu / self.r0**2)
#             beta = 0
#             f = self.f0 + beta * (y - L/2)

#             # wind forcing, bottom drag
#             tau0 = 0.
#             bottom_drag_coef = 0.

#             apply_mask = False # True

#             # create rankine vortex in PV
#             xc = 0.5 * (xv[1:] + xv[:-1])
#             yc = 0.5 * (yv[1:] + yv[:-1])
#             x, y = torch.meshgrid(xc, yc, indexing='ij')
#             r = torch.sqrt(x**2 + y**2)
#             # circular domain mask
#             mask = (r < L/2).type(torch.float64) if apply_mask else torch.ones_like(x)

#             param = {
#                 'nx': self.nx,
#                 'ny': self.ny,
#                 'nl': nl,
#                 'mask': mask,
#                 'n_ens': 1,
#                 'Lx': L,
#                 'Ly': L,
#                 'flux_stencil': 5,
#                 'H': H,
#                 'g_prime': g_prime,
#                 'tau0': tau0,
#                 'f0': self.f0,
#                 'beta': beta,
#                 'bottom_drag_coef': bottom_drag_coef,
#                 'device': device,
#                 'dt': 0, # time-step (s)
#             }
#             self.qg = QGFV(param)

#     def forward_QG(self, x):
#         # take as input (in the case of batch size ==1) a tensor x of size [B=1,D,H,W] with B the size of batch, D the lenght of data assimilation window (typically 15 days), H and W the spatial size
#         # return a vector of the same size with x[:,j+1,:,:] = QG(x[:,j,:,:])

#         ## example for a single prediction x[:,j+1,:,:] = QG(x[:,j,:,:]), will need to make a loop later
#         ssh_0 = x[0][0].T
#         ssh_1 = self.forecast_QG(ssh_0).T       

#         return ssh_1
    
#     def forward(self, state):
#         #return F.mse_loss(state, self.forward_QG(state)) 
#         return F.mse_loss(state[0][0], self.forward_QG(state)) 
    
#     def forecast_QG(self, ssh_0):
#         with torch.no_grad():
#             ## initialize psi_0 from ssh map
#             ssh_0 = ssh_0
#             psi_2d = (self.g/self.f0) * ssh_0
#             self.qg.psi = psi_2d.unsqueeze(0).unsqueeze(0)

#             ## initialize q_0 from psi_0
#             self.qg.compute_q_from_psi()

#             # set amplitude to have correct Rossby number
#             u, v = self.qg.grad_perp(self.qg.psi, self.qg.dx, self.qg.dy)
#             u_norm_max = max(torch.abs(u).max().item(), torch.abs(v).max().item())
#             factor = self.Ro * self.f0 * self.r0 / u_norm_max
#             self.qg.psi *= factor
#             self.qg.q *= factor
#             u, v = self.qg.grad_perp(self.qg.psi, self.qg.dx, self.qg.dy)
#             u_max = u.max().cpu().item()
#             v_max = v.max().cpu().item()
#             #print(f'u_max {u_max:.2e}, v_max {v_max:.2e}')

#             # set time step with CFL
#             dt = self.cfl * min(self.dx / u_max, self.dy / v_max)
#             self.qg.dt = dt
#             #print(f'integration time step : {round(dt)//3600} hour(s), {(round(dt)% 3600) // 60} minute(s), and {(round(dt)% 3600) % 60} second(s). ')

#             # time params
#             t = 0
#             t_end = round(1 * 86400 ) # 1 day expressed in sec
#             n_steps = int(t_end / dt) + 1 

#             # plot, log, check nans
#             freq_plot = int(t_end / 25 / dt) + 1
#             freq_log = int(t_end / 50 / dt) + 1

#             t0 = time.time()
#             for n in range(1, n_steps+1): #n_steps+1): #100
#                 self.qg.step()
#                 t += dt

#             psi_forecast = self.qg.psi * (1/factor) # rescale ssh as input
#             ssh_forecast = psi_forecast * (self.f0/self.g)
#             ssh_forecast = ssh_forecast[0][0] # pass from [1,1,240,240] (one layer QG) -> [240,240]
#             return ssh_forecast
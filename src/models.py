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

        ### Initialisation of state with MDT ###
        lon_min = -64 
        lon_max = -52.
        lat_max = 44.
        lat_min = 32.
        mdt = torch.tensor(xr.open_dataset('/DATASET/2023_SSH_mapping_train_eNATL60_test_NATL60/NATL60-CJM165/ds_ref_1_20.nc')['mdt']
                .sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.astype(np.float32))
        mdt = (mdt - 0.3174050238130743) / 0.3889927646359018
        mdt_tensor = mdt.to('cuda').unsqueeze(0).unsqueeze(1).repeat(1, 10, 1, 1)
        self.mdt = torch.nan_to_num(mdt_tensor, nan=0.0)

        ### Initialisation of state with saptially filtered GT data ###
        # Function to create a Gaussian kernel
        def gaussian_kernel(size, sigma):
            x = torch.arange(-size//2 + 1., size//2 + 1.)
            y = torch.arange(-size//2 + 1., size//2 + 1.)
            x_grid, y_grid = torch.meshgrid(x, y)
            kernel = torch.exp(-0.5 * (x_grid**2 + y_grid**2) / sigma**2)
            kernel = kernel / kernel.sum()
            return kernel.to('cuda')
        # Create a Gaussian kernel
        self.kernel_size = 21
        self.sigma = 8
        gaussian_kernel = gaussian_kernel(self.kernel_size, self.sigma)
        self.gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)  # Reshape for 2D convolution
        self.padding_size = self.kernel_size // 2

    
    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        ### Initialisation of state with MDT ###
        #return self.mdt.detach().requires_grad_(True)

        ## Initialisation of state with saptially filtered GT data ###

        # Replace NaNs in the tensor with 0
        gt = torch.nan_to_num(batch.tgt, nan=0.0).to('cuda')

        # Apply Gaussian filter to each 2D slice along dimension 1
        smoothed_out_data = torch.empty_like(gt)
        self.padding_size = self.kernel_size // 2
        for i in range(gt.shape[1]):
            # Apply reflection padding before convolution
            padded_slice = F.pad(gt[0, i].unsqueeze(0).unsqueeze(0), (self.padding_size, self.padding_size, self.padding_size, self.padding_size), mode='reflect')
            smoothed_out_data[0, i] = F.conv2d(padded_slice, self.gaussian_kernel).squeeze()

        x_init = smoothed_out_data.detach().requires_grad_(True)

        # return batch.input.nan_to_num().detach().requires_grad_(True)
        # return torch.zeros_like(batch.input).detach().requires_grad_(True)

        #x_init = torch.nan_to_num(batch.tgt, nan=0.0).detach().requires_grad_(True)
        torch.save(x_init,"/homes/g24meda/lab/4dvarnet-starter/outputs/init_state.pt")
        torch.save(torch.nan_to_num(batch.tgt, nan=0.0).to('cuda'),"/homes/g24meda/lab/4dvarnet-starter/outputs/gt.pt")
        return x_init
    

    def solver_step(self, state, batch, step):
        ## add regularization 
        prior_grad = kfilts.spatial_gradient(self.prior_cost.forward_QG(state).unsqueeze(0), normalized=False)
        # Calculate the L2 norm of the gradient
        l2_norm_grad = torch.norm(prior_grad, p=2)
        ##
        alpha_1 = 1
        alpha_2 = 1
        var_cost = alpha_1 * self.prior_cost(state) + alpha_2 * self.obs_cost(state, batch) #+ 0.0001 * l2_norm_grad
        print('prior cost', alpha_1 * self.prior_cost(state))
        print('obs cost', alpha_2 *self.obs_cost(state, batch))
        print('grad prior norm', l2_norm_grad)

        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        # state_update = (
        #     1 / (step + 1) * gmod
        #         + self.lr_grad * (step + 1) / self.n_step * grad
        # )

        state_update = (
            self.lr_grad * grad #* (step + 1) / self.n_step * grad
        )
        
        torch.save(state,'/homes/g24meda/lab/4dvarnet-starter/outputs/reconstructed_state.pt')
        torch.save(batch.tgt,'/homes/g24meda/lab/4dvarnet-starter/outputs/target.pt')
        
        ## debugging
        # if torch.isnan(state_update).any():
        #     print("NaN detected in state_update.")
        #     if torch.isnan(state).any():
        #         print("NaN detected in state.")
        #     else :
        #         print("No NaN detected in state.")
        #     torch.save(state,'/homes/g24meda/lab/4dvarnet-starter/outputs/reconstructed_state.pt')
        #     torch.save(batch.tgt,'/homes/g24meda/lab/4dvarnet-starter/outputs/target.pt')

        #     if torch.isnan(var_cost).any():
        #         print("NaN detected in var_cost.")
        #     #torch.save(var_cost,'/homes/g24meda/lab/4dvarnet-starter/outputs/var_cost.pt')

        #     if torch.isnan(grad).any():
        #         print("NaN detected in grad.")
        #     torch.save(grad,'/homes/g24meda/lab/4dvarnet-starter/outputs/grad.pt')

        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            #torch.save(state,'/homes/g24meda/lab/4dvarnet-starter/outputs/init_state.pt')
            self.grad_mod.reset_state(batch.input)
            print('\n'+'### new batch ###'+ '\n')
            i = 0
            for step in range(self.n_step):
                print('\n## solver iteration n°: '+ str(i) + ' ## \n')
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)
                i += 1 

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
        torch.save(state,'/homes/g24meda/lab/4dvarnet-starter/outputs/state_obs_cost.pt')
        torch.save(batch.input.nan_to_num(),'/homes/g24meda/lab/4dvarnet-starter/outputs/gt_obs_cost.pt')
        torch.save(state[msk],'/homes/g24meda/lab/4dvarnet-starter/outputs/masked_state_obs_cost.pt')
        torch.save(batch.input.nan_to_num()[msk],'/homes/g24meda/lab/4dvarnet-starter/outputs/masked_gt_obs_cost.pt')
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
    def __init__(self, nl, L, g_prime, H, f0, beta, tau0, bottom_drag_coef, x_dim, y_dim, apply_mask=False) -> None:
        super().__init__()

        print('initialisation of QG')
        ## QGCost class attributes
        self.dtype = torch.float64
        self.g = 10.0  # m/s^2
        dt = 60 *10 #60 * 40  # integration time step : 10 min
        self.t_end = round(1 * 86400)  # time of intergation for each QG forecast : 1 day expressed in sec

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        nx = x_dim - 1
        ny = y_dim - 1
        # dx = L / nx
        # dy = L / ny
        xv = torch.linspace(-L / 2, L / 2, nx+1 , dtype=self.dtype, device=device)
        yv = torch.linspace(-L / 2, L / 2, ny+1, dtype=self.dtype, device=device)
        x, y = torch.meshgrid(xv, yv, indexing='ij')

        H_tensor = torch.zeros(nl, 1, 1, dtype=self.dtype, device=device)
        if nl == 1:
            H_tensor[0, 0, 0] = H

        g_prime_tensor = torch.zeros(nl, 1, 1, dtype=self.dtype, device=device)
        if nl == 1:
            g_prime_tensor[0, 0, 0] = g_prime

        f = f0 + beta * (y - L / 2)

        # create rankine vortex in PV
        xc = 0.5 * (xv[1:] + xv[:-1])
        yc = 0.5 * (yv[1:] + yv[:-1])
        x, y = torch.meshgrid(xc, yc, indexing='ij')
        r = torch.sqrt(x**2 + y**2)

        mask = torch.ones_like(x, dtype=torch.float64)
        if apply_mask:
            dist_to_left = x - x.min()
            dist_to_right = x.max() - x
            dist_to_bottom = y - y.min()
            dist_to_top = y.max() - y
            dist_to_boundary = torch.min(torch.min(dist_to_left, dist_to_right), torch.min(dist_to_bottom, dist_to_top))
            soft_step = lambda x: torch.sigmoid(x / 10)
            normalized_dist = dist_to_boundary / dist_to_boundary.max() * 100
            self.mask = soft_step(normalized_dist).type(torch.float64)


        param = {
        'nx': x_dim - 1,
        'ny': y_dim - 1,
        'nl': nl,
        'mask': mask,
        'n_ens': 1,
        'Lx': L,
        'Ly': L,
        'flux_stencil': 5,
        'H': H_tensor,
        'g_prime': g_prime_tensor,
        'tau0': tau0,
        'f0': f0,
        'beta': beta,
        'bottom_drag_coef': bottom_drag_coef,
        'device': device,
        'dt': dt,
        'fixed_psi_boundary': None,
        'fixed_q_boundary': None
        }
        
        self.qg = QGFV(param)

    def forward_QG(self, x):
        B, D, H, W = x.shape
        if B != 1:
            raise Exception("4dvarnet-QG does not handle batch with size > 1. Please set batch size to 1.")
        
        forecasts = torch.zeros_like(x)
        for i in range(D - 1):
            ssh_0 = x[0][i].T
            ssh_1 = self.forecast_QG(ssh_0).T
            forecasts[0][i + 1] = ssh_1

        forecasts[0][0] = x[0][0]
        return forecasts[0][:]

    def forward(self, state):
        return F.mse_loss(state[0][1:], self.forward_QG(state)[1:])

    def forecast_QG(self, ssh_0):
        with torch.no_grad():
            torch.backends.cudnn.deterministic = True

            # initialize psi_0 with ssh_0 input
            psi_2d = (self.g / self.qg.f0) * ssh_0
            self.qg.psi = psi_2d.unsqueeze(0).unsqueeze(0)
            self.qg.compute_q_from_psi()
            self.qg.fixed_psi_boundary = self.qg.psi.clone().detach()
            self.qg.fixed_q_boundary = self.qg.q.clone().detach()

            t = 0
            n_steps = int(self.t_end / self.qg.dt) + 1

            for n in range(1, n_steps + 1):
                self.qg.step()
                t += self.qg.dt

            if torch.isnan(self.qg.q).any():
                print('NaN found in q')

            if torch.isnan(self.qg.psi).any():
                print('NaN found in psi')
                raise ValueError("NaN values found in qg.psi")

            ssh_forecast = self.qg.psi * (self.qg.f0 / self.g)
            ssh_forecast = ssh_forecast[0][0]
            return ssh_forecast

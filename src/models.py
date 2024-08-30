import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from scipy.ndimage import distance_transform_edt
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
        training_loss = 50.0 * loss + 1000.0 * grad_loss + 1.0 * prior_cost
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

class GradSolver_Id(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_mod,lr_grad=0.2, save_debugg_path = None , **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_mod = lr_mod
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
        mdt_tensor = mdt.to('cuda').unsqueeze(0).unsqueeze(1).repeat(1, 15, 1, 1)
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

        # monitor the variationnal cost
        self.save_debugg_path = save_debugg_path
        if self.save_debugg_path is not None:
            self.prior_cost_values = []
            self.obs_cost_values = []
            self.var_cost_values = []
            self.background_values = []

            self.prior_cost_values_current = []
            self.obs_cost_values_current = []
            self.var_cost_values_current = []

            if not os.path.exists(self.save_debugg_path):
                os.makedirs(self.save_debugg_path)
                
    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        ## Initialisation of state with MDT ###
        x_init = self.mdt.detach().requires_grad_(True)
        
        if self.save_debugg_path is not None:
            torch.save(x_init, self.save_debugg_path + 'init_state.pt')
        return x_init
    

    def solver_step(self, state, batch, step):
      
        alpha_1 = 1.0 #150.0 #1.0 
        alpha_2 = 0.5 #5.0 #0.5
        var_cost = alpha_1 * self.prior_cost(state) + alpha_2 * self.obs_cost(state, batch) 

        # Store costs for all batches
        prior_cost_value = alpha_1 * self.prior_cost(state).item()
        obs_cost_value = alpha_2 * self.obs_cost(state, batch).item()
        var_cost_value = var_cost.item()
        self.prior_cost_values.append(prior_cost_value)
        self.obs_cost_values.append(obs_cost_value)
        self.var_cost_values.append(var_cost_value)

        # Store costs for current batch
        if self.save_debugg_path is not None:
            if step == 0:
                self.prior_cost_values_current = []
                self.obs_cost_values_current = []
                self.var_cost_values_current = []
            self.prior_cost_values_current.append(prior_cost_value)
            self.obs_cost_values_current.append(obs_cost_value)
            self.var_cost_values_current.append(var_cost_value)
            # Save figure of costs curves
            #if step % 1 == 0:
            if step == (self.n_step - 1):
                # Store costs for all batches
                plt.figure(figsize=(10, 5))
                plt.plot(self.prior_cost_values, label='Prior Cost')
                plt.plot(self.obs_cost_values, label='Observation Cost')
                plt.plot(self.var_cost_values, label='Total Cost')
                plt.xlabel('Solver iteration')
                plt.ylabel('Cost')
                plt.title('Costs evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(self.save_debugg_path + 'var_costs_all_batches.png')
                plt.close()

                # Store costs for current batch
                plt.figure(figsize=(10, 5))
                plt.plot(self.prior_cost_values_current, label='Prior Cost')
                plt.plot(self.obs_cost_values_current, label='Observation Cost')
                plt.plot(self.var_cost_values_current, label='Total Cost')
                plt.xlabel('Solver iteration')
                plt.ylabel('Cost')
                plt.title('Costs evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(self.save_debugg_path + 'var_costs_current_batch.png')
                plt.close()

        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
    
        state_update = (
            self.lr_mod * 1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )

        ### Apply gaussian kernel to the update ##
        
        def gaussian_kernel_update(size: int, sigma: float):
            """Create a 2D Gaussian kernel."""
            x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
            y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
            xx, yy = torch.meshgrid(x, y)
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            return kernel

        def apply_gaussian_smoothing(tensor: torch.Tensor, kernel_size: int, sigma: float):
            """Apply Gaussian smoothing to the last two dimensions of a 4D tensor."""
            N, C, H, W = tensor.shape
            
            # Create Gaussian kernel
            kernel = gaussian_kernel_update(kernel_size, sigma)
            kernel = kernel.view(1, 1, kernel_size, kernel_size).to(tensor.device)
            
            # Duplicate the kernel for each channel
            kernel = kernel.repeat(C, 1, 1, 1)
            
            # Apply the Gaussian kernel to each channel
            padding = kernel_size // 2
            smoothed_tensor = F.conv2d(tensor, kernel, padding=padding, groups=C)
            
            return smoothed_tensor

        kernel_size = 21  # Taille du noyau
        sigma = 2.0 #2.0       # Écart-type

        smoothed_update = apply_gaussian_smoothing(state_update, kernel_size, sigma)

        if self.save_debugg_path is not None:
            print('gmod coeff', self.lr_mod * 1 / (step + 1))
            print('norm of gmod update', torch.norm(self.lr_mod * 1 / (step + 1) * gmod, p=2))
            print('grad coeff', self.lr_grad * (step + 1) / self.n_step)
            print('norm of grad update', torch.norm(self.lr_grad *(step + 1) / self.n_step * grad, p=2))

            torch.save(state, self.save_debugg_path + 'state.pt')
            torch.save(state_update, self.save_debugg_path + 'state_update.pt')
            torch.save(smoothed_update, self.save_debugg_path + 'smoothed_update.pt')
            torch.save(batch.tgt, self.save_debugg_path + 'target.pt')
        
        # debugging
        if torch.isnan(state_update).any():
            raise ValueError("NaN detected in state_update, saving last clean state")
      
        return state - smoothed_update # state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)
            print('\n'+'### new batch ###'+ '\n')
            for step in range(self.n_step):
                if self.save_debugg_path is not None:
                    print('\n## solver iteration n°: '+ str(step) + ' ## \n')
                state = self.solver_step(state, batch, step=step)

                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_id(state)
        return state
    
    
class GradSolver_QG(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_mod,lr_grad=0.2, save_debugg_path = None,**kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_mod = lr_mod
        self.lr_grad = lr_grad

        self._grad_norm = None

        # ==============================================
        #      Initialisation of state with MDT
        # ==============================================
        lon_min = -64 
        lon_max = -52.
        lat_max = 44.
        lat_min = 32.
        mdt = torch.tensor(xr.open_dataset('/DATASET/2023_SSH_mapping_train_eNATL60_test_NATL60/NATL60-CJM165/ds_ref_1_20.nc')['mdt']
                .sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).values.astype(np.float32))
        mdt = (mdt - 0.3174050238130743) / 0.3889927646359018
        mdt_tensor = mdt.to('cuda').unsqueeze(0).unsqueeze(1).repeat(1, 15, 1, 1)
        self.mdt = torch.nan_to_num(mdt_tensor, nan=0.0)

        # ==============================================
        #     Monitoring of the variationnal costs
        # ==============================================
        self.save_debugg_path = save_debugg_path
        if self.save_debugg_path is not None:
            self.prior_cost_values = []
            self.obs_cost_values = []
            self.var_cost_values = []
            self.background_values = []

            self.prior_cost_values_current = []
            self.obs_cost_values_current = []
            self.var_cost_values_current = []

            if not os.path.exists(self.save_debugg_path):
                os.makedirs(self.save_debugg_path)

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        # ==============================================
        #      Initialisation of state with MDT
        # ==============================================
        x_init = self.mdt.detach().requires_grad_(True)

        # ==============================================
        #      Initialisation of state with obs 
        # ==============================================
        # x_init = batch.input.nan_to_num().detach().requires_grad_(True) 
        
        if self.save_debugg_path is not None:
            torch.save(x_init, self.save_debugg_path + 'init_state.pt')
        return x_init
    

    def solver_step(self, state, batch, step):      
        alpha_1 = 1.0  
        alpha_2 = 0.5 
        var_cost = alpha_1 * self.prior_cost(state) + alpha_2 * self.obs_cost(state, batch) 

        # ==============================================
        #     Monitoring of the variationnal costs
        # ==============================================

        if self.save_debugg_path is not None:
            # Store costs for all batches
            prior_cost_value = alpha_1 * self.prior_cost(state).item()
            obs_cost_value = alpha_2 * self.obs_cost(state, batch).item()
            var_cost_value = var_cost.item()
            self.prior_cost_values.append(prior_cost_value)
            self.obs_cost_values.append(obs_cost_value)
            self.var_cost_values.append(var_cost_value)

            # Store costs for current batch
            if step == 0:
                self.prior_cost_values_current = []
                self.obs_cost_values_current = []
                self.var_cost_values_current = []
            self.prior_cost_values_current.append(prior_cost_value)
            self.obs_cost_values_current.append(obs_cost_value)
            self.var_cost_values_current.append(var_cost_value)


            # Save figure of costs curves
            if step == (self.n_step - 1):
                # Store costs for all batches
                plt.figure(figsize=(10, 5))
                plt.plot(self.prior_cost_values, label='Prior Cost')
                plt.plot(self.obs_cost_values, label='Observation Cost')
                plt.plot(self.var_cost_values, label='Total Cost')
                plt.xlabel('Solver iteration')
                plt.ylabel('Cost')
                plt.title('Costs evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(self.save_debugg_path + 'var_costs_all_batches.png')
                plt.close()

                # Store costs for current batch
                plt.figure(figsize=(10, 5))
                plt.plot(self.prior_cost_values_current, label='Prior Cost')
                plt.plot(self.obs_cost_values_current, label='Observation Cost')
                plt.plot(self.var_cost_values_current, label='Total Cost')
                plt.xlabel('Solver iteration')
                plt.ylabel('Cost')
                plt.title('Costs evolution')
                plt.legend()
                plt.grid(True)
                plt.savefig(self.save_debugg_path + 'var_costs_current_batch.png')
                plt.close()

        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        gmod = self.grad_mod(grad)
        # state_update = (
        #     1 / (step + 1) * gmod
        #         + self.lr_grad * (step + 1) / self.n_step * grad
        # )
        state_update = (
            self.lr_mod * 1 / (step + 1) * gmod
                + self.lr_grad * (step + 1) / self.n_step * grad
        )

        # ==============================================
        #      Apply gaussian kernel to the update 
        # ==============================================

        def gaussian_kernel_update(size: int, sigma: float):
            """Create a 2D Gaussian kernel."""
            x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
            y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
            xx, yy = torch.meshgrid(x, y)
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            return kernel

        def apply_gaussian_smoothing(tensor: torch.Tensor, kernel_size: int, sigma: float):
            """Apply Gaussian smoothing to the last two dimensions of a 4D tensor."""
            N, C, H, W = tensor.shape
            
            # Create Gaussian kernel
            kernel = gaussian_kernel_update(kernel_size, sigma)
            kernel = kernel.view(1, 1, kernel_size, kernel_size).to(tensor.device)
            
            # Duplicate the kernel for each channel
            kernel = kernel.repeat(C, 1, 1, 1)
            
            # Apply the Gaussian kernel to each channel
            padding = kernel_size // 2
            smoothed_tensor = F.conv2d(tensor, kernel, padding=padding, groups=C)
            
            return smoothed_tensor

        # Initialize the gaussian kernel
        kernel_size = 21  
        sigma = 2.0 

        # Apply it to the update
        smoothed_update = apply_gaussian_smoothing(state_update, kernel_size, sigma)

        if self.save_debugg_path is not None:
            print('gmod coeff', self.lr_mod * 1 / (step + 1))
            print('norm of gmod update', torch.norm(self.lr_mod * 1 / (step + 1) * gmod, p=2))
            print('grad coeff', self.lr_grad * (step + 1) / self.n_step)
            print('norm of grad update', torch.norm(self.lr_grad *(step + 1) / self.n_step * grad, p=2))

            torch.save(state, self.save_debugg_path + 'state.pt')
            torch.save(state_update, self.save_debugg_path + 'state_update.pt')
            torch.save(smoothed_update, self.save_debugg_path + 'smoothed_update.pt')
            torch.save(batch.tgt, self.save_debugg_path + 'target.pt')
        
        # debugging
        if torch.isnan(state_update).any():
            raise ValueError("NaN detected in state_update, saving last clean state")
        
        return state - smoothed_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)
            print('\n'+'### new batch ###'+ '\n')
            for step in range(self.n_step):
                if self.save_debugg_path is not None:
                    print('\n## solver iteration n°: '+ str(step) + ' ## \n')
                state = self.solver_step(state, batch, step=step)

                if not self.training:
                    state = state.detach().requires_grad_(True)

            if not self.training:
                state = self.prior_cost.forward_QG(state).unsqueeze(0)
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


# ======================================================================
#     BilinAE Prior Cost : classic prior cost in 4dvarnet-starter conf 
# ======================================================================

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

# ==================================================
#     Identity Prior Cost : prior phi is identity 
# ==================================================

class Cost_Id(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward_id(self, x):
        return x
    
    def forward(self, state):
        return F.mse_loss(state, self.forward_id(state)) 
        

# ==============================================
#     Usefull functions to run the QG model 
# ==============================================

def gaspari_cohn(r, c):
    """
    Gaspari-Cohn function. Inspired from E.Cosmes.
        
    Args: 
        r : array of value whose the Gaspari-Cohn function will be applied
        c : Distance above which the return values are zeros

    Returns:  smoothed values 
    """ 
    if isinstance(r, (float, int)):
        ra = torch.tensor([r], dtype=torch.float)
    else:
        ra = torch.tensor(r, dtype=torch.float)
        
    if c <= 0:
        return torch.zeros_like(ra)
    else:
        ra = 2 * torch.abs(ra) / c
        gp = torch.zeros_like(ra)
        
        # Conditions for the Gaspari-Cohn function
        i1 = ra <= 1.
        i2 = (ra > 1.) & (ra <= 2.)
        
        # Applying the Gaspari-Cohn function
        gp[i1] = -0.25 * ra[i1]**5 + 0.5 * ra[i1]**4 + 0.625 * ra[i1]**3 - (5./3.) * ra[i1]**2 + 1.
        gp[i2] = (1./12.) * ra[i2]**5 - 0.5 * ra[i2]**4 + 0.625 * ra[i2]**3 + (5./3.) * ra[i2]**2 - 5. * ra[i2] + 4. - (2./3.) / ra[i2]
        
        if isinstance(r, float):
            gp = gp.item()
            
    return gp

def lonlat2dxdy(lon,lat):
    dlon = torch.gradient(lon)
    dlat = torch.gradient(lat)
    dx = torch.sqrt((dlon[1]*111000*torch.cos(torch.deg2rad(lat)))**2
                 + (dlat[1]*111000)**2)
    dy = torch.sqrt((dlon[0]*111000*torch.cos(torch.deg2rad(lat)))**2
                 + (dlat[0]*111000)**2)
    dx[0,:] = dx[1,:]
    dx[-1,: ]= dx[-2,:] 
    dx[:,0] = dx[:,1]
    dx[:,-1] = dx[:,-2]
    dy[0,:] = dy[1,:]
    dy[-1,:] = dy[-2,:] 
    dy[:,0] = dy[:,1]
    dy[:,-1] = dy[:,-2]
    
    return dx,dy

def dstI1D(x, norm='ortho'):
    """1D type-I discrete sine transform."""
    return torch.fft.irfft(-1j * torch.nn.functional.pad(x, (1, 1)), norm=norm)[...,1:x.shape[-1]+1]

def dstI2D(x, norm='ortho'):
    """2D type-I discrete sine transform."""
    return dstI1D(dstI1D(x, norm=norm).transpose(-1, -2), norm=norm).transpose(-1, -2)

def inverse_elliptic_dst(f, operator_dst):
    """Inverse elliptic operator (e.g. Laplace, Helmoltz)
    using float32 discrete sine transform."""
    return dstI2D(dstI2D(f) / operator_dst)


# ======================================================================================
#    QG Prior Cost : new cost integrating the QG model
# ======================================================================================
# Description:
#    The QG Prior Cost represents a new prior cost that integrates
#    the Quasi-Geostrophic (QG) model.
#
# Formula:
#    |x - phi_qg(x)|
#
# Details:
#    - phi_qg(x) corresponds to N independent forward integrations
#      of the QG model over a time span of 1 day each.
#    - N is typically 15, representing a temporal batch size of 15 days.
#    - Each integration has a time step of typiccaly < 10 minutes.
#    - Specifically:
#        phi_qg(x)(t+1) = forward_qg(x(t))
#        ie phi_qg : x -> [forward_qg(x(0)),forward_qg(x(1)), .. , forward_qg(x(N-1))]
#    - Here, forward_qg(x(t)) denotes the result of a 1-day forward integration
#      of the QG model starting from day t to day t+1.
# =======================================================================================

class QGCost_new(nn.Module):

    ###########################################################################
    #                             Initialization                              #
    ###########################################################################
    def __init__(self, domain_limits=None, res=0.05, avg_pool = None, dt=None, tint=None, SSH=None, c=None, g=9.81, f=None, Kdiffus=None, device='cpu',save_debugg_path = None, mean=None, std=None):
        super().__init__()
        print('initialisation of QG')
        
        # average pooling before QG forward (reduce dx and dt => computation time)
        self.avg_pool = avg_pool
        if self.avg_pool is not None:
            res = res * self.avg_pool
        
        # Coordinates
        lon = torch.arange(domain_limits['lon'].start, domain_limits['lon'].stop , res, dtype=torch.float64)
        lat = torch.arange(domain_limits['lat'].start, domain_limits['lat'].stop , res, dtype=torch.float64)

        if len(lon.shape)==1:
            lon,lat = torch.meshgrid(lon,lat) 
        dx,dy = lonlat2dxdy(lon,lat)

        # Grid shapeet y
        ny, nx = dx.shape
        self.nx = nx
        self.ny = ny

        # Grid spacing
        dx = dy = (torch.nanmean(dx) + torch.nanmean(dy)) / 2
        self.dx = dx
        self.dy = dy
        
        # Time step
        self.dt = dt
        
        # Integration time (1 day)
        self.tint = tint

        # Gravity
        self.g = torch.tensor(g).to(device=device, dtype=torch.float)

        # Coriolis
        if hasattr(f, "__len__"):
            self.f = (torch.nanmean(torch.tensor(f)) * torch.ones((self.ny, self.nx)))
        elif f is not None:
            self.f = (f * torch.ones((self.ny, self.nx)))
        else:
            self.f = 4*torch.pi/86164*torch.sin(lat*torch.pi/180)
        self.f = self.f.to(device=device, dtype=torch.float)

        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = (torch.nanmean(torch.tensor(c)) * torch.ones((self.ny, self.nx))).to(device=device, dtype=torch.float)
        else:
            self.c = (c * torch.ones((self.ny, self.nx))).to(device=device, dtype=torch.float)

        # Elliptical inversion operator
        x, y = torch.meshgrid(torch.arange(1, nx - 1, dtype=torch.float),
                              torch.arange(1, ny - 1, dtype=torch.float))
        x = x.to(device=device, dtype=torch.float)
        y = y.to(device=device, dtype=torch.float)
        laplace_dst = 2 * (torch.cos(torch.pi / (nx - 1) * x) - 1) / self.dx ** 2 + \
                      2 * (torch.cos(torch.pi / (ny - 1) * y) - 1) / self.dy ** 2
        self.helmoltz_dst = self.g / self.f.mean() * laplace_dst - self.g * self.f.mean() / self.c.mean() ** 2

        # get land pixels
        if SSH is not None:
            isNAN = torch.isnan(SSH).to(device=device, dtype=torch.bool)
        else:
            isNAN = None

        ################
        # Mask array
        ################

        # mask=3 away from the coasts
        mask = torch.zeros((ny, nx), dtype=torch.int) + 3

        # mask=1 for borders of the domain 
        mask[0, :] = 1
        mask[:, 0] = 1
        mask[-1, :] = 1
        mask[:, -1] = 1

        # mask=2 for pixels adjacent to the borders 
        mask[1, 1:-1] = 2
        mask[1:-1, 1] = 2
        mask[-2, 1:-1] = 2
        mask[-3, 1:-1] = 2
        mask[1:-1, -2] = 2
        mask[1:-1, -3] = 2

        # mask=0 on land 
        if isNAN is not None:
            mask[isNAN] = 0.
            indNan = torch.argwhere(isNAN)
            for i, j in indNan:
                for p1 in range(-2, 3):
                    for p2 in range(-2, 3):
                        itest = i + p1
                        jtest = j + p2
                        if ((itest >= 0) & (itest <= ny - 1) & (jtest >= 0) & (jtest <= nx - 1)):
                            # mask=1 for coast pixels
                            if (mask[itest, jtest] >= 2) and (p1 in [-1, 0, 1] and p2 in [-1, 0, 1]):
                                mask[itest, jtest] = 1
                            # mask=1 for pixels adjacent to the coast
                            elif (mask[itest, jtest] == 3):
                                mask[itest, jtest] = 2

        self.mask = mask.to(device=device, dtype=torch.int)
        self.ind0 = (mask == 0).to(device=device, dtype=torch.bool)
        self.ind1 = (mask == 1).to(device=device, dtype=torch.bool)
        self.ind2 = (mask == 2).to(device=device, dtype=torch.bool)
        self.ind12 = (self.ind1 + self.ind2).to(device=device, dtype=torch.bool)

        # Diffusion coefficient 
        self.Kdiffus = Kdiffus

        # save debugg_path
        self.save_debugg_path = save_debugg_path

        # unormalization
        self.mean = mean
        self.std = std

    def h2uv(self, h):
        """ SSH to U,V

        Args:
            h (2D array): SSH field.

        Returns:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
        """
    
        u = torch.zeros_like(h)
        v = torch.zeros_like(h)

        u[..., 1:-1, 1:] = - self.g / self.f[None, 1:-1, 1:] * (h[..., 2:, :-1] + h[..., 2:, 1:] - h[..., :-2, 1:] - h[..., :-2, :-1]) / (4 * self.dy)
        v[..., 1:, 1:-1] = self.g / self.f[None, 1:, 1:-1] * (h[..., 1:, 2:] + h[..., :-1, 2:] - h[..., :-1, :-2] - h[..., 1:, :-2]) / (4 * self.dx)
        
        u = torch.where(torch.isnan(u), torch.tensor(0.0), u)
        v = torch.where(torch.isnan(v), torch.tensor(0.0), v)
            
        return u, v

    def h2pv(self, h, hbc, c=None):
        """ SSH to Q

        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius

        Returns:
            q: Potential Vorticity field
        """
        
        if c is None:
            c = self.c

        q = torch.zeros_like(h, dtype=torch.float)

        q[..., 1:-1, 1:-1] = (
            self.g / self.f[None, 1:-1, 1:-1] * 
            ((h[..., 2:, 1:-1] + h[..., :-2, 1:-1] - 2 * h[..., 1:-1, 1:-1]) / self.dy ** 2 +
             (h[..., 1:-1, 2:] + h[..., 1:-1, :-2] - 2 * h[..., 1:-1, 1:-1]) / self.dx ** 2) - 
            self.g * self.f[None, 1:-1, 1:-1] / (c[None, 1:-1, 1:-1] ** 2) * h[..., 1:-1, 1:-1]
        )

        q = torch.where(torch.isnan(q), torch.tensor(0.0), q)
        q[..., self.ind12] = - self.g * self.f[None,self.ind12] / (c[None,self.ind12] ** 2) * hbc[...,self.ind12]
        q[..., self.ind0] = 0

        return q

    def rhs(self, u, v, q0, way=1):
        """ increment

        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)

        Returns:
            rhs (2D array): advection increment
        """

        # Upwind current
        u_on_T = way * 0.5 * (u[..., 1:-1, 1:-1] + u[..., 1:-1, 2:])
        v_on_T = way * 0.5 * (v[..., 1:-1, 1:-1] + v[..., 2:, 1:-1])
        up = torch.where(u_on_T < 0, torch.tensor(0.0), u_on_T)
        um = torch.where(u_on_T > 0, torch.tensor(0.0), u_on_T)
        vp = torch.where(v_on_T < 0, torch.tensor(0.0), v_on_T)
        vm = torch.where(v_on_T > 0, torch.tensor(0.0), v_on_T)

        # PV advection
        rhs_q = self._adv(up, vp, um, vm, q0)
        rhs_q[..., 2:-2, 2:-2] -= way * (self.f[None, 3:-1, 2:-2] - self.f[None, 1:-3, 2:-2]) / (2 * self.dy) * 0.5 * (v[..., 2:-2, 2:-2] + v[..., 3:-1, 2:-2])
        
        # PV Diffusion
        if self.Kdiffus is not None:
            rhs_q[..., 2:-2, 2:-2] += (
                self.Kdiffus / (self.dx ** 2) * (q0[..., 2:-2, 3:-1] + q0[..., 2:-2, 1:-3] - 2 * q0[..., 2:-2, 2:-2]) +
                self.Kdiffus / (self.dy ** 2) * (q0[..., 3:-1, 2:-2] + q0[..., 1:-3, 2:-2] - 2 * q0[..., 2:-2, 2:-2])
            )
        rhs_q = torch.where(torch.isnan(rhs_q), torch.tensor(0.0), rhs_q)
        rhs_q[..., self.ind12] = 0
        rhs_q[..., self.ind0] = 0

        return rhs_q

    def _adv(self, up, vp, um, vm, var0):
        """
            3rd-order upwind scheme.
        """

        res = torch.zeros_like(var0, dtype=torch.float)

        res[..., 2:-2,2:-2] = \
            - up[..., 1:-1, 1:-1] * 1 / (6 * self.dx) * \
            (2 * var0[..., 2:-2, 3:-1] + 3 * var0[..., 2:-2, 2:-2] - 6 * var0[..., 2:-2, 1:-3] + var0[..., 2:-2, :-4]) \
            + um[..., 1:-1, 1:-1] * 1 / (6 * self.dx) * \
            (var0[..., 2:-2, 4:] - 6 * var0[..., 2:-2, 3:-1] + 3 * var0[..., 2:-2, 2:-2] + 2 * var0[..., 2:-2, 1:-3]) \
            - vp[..., 1:-1, 1:-1] * 1 / (6 * self.dy) * \
            (2 * var0[..., 3:-1, 2:-2] + 3 * var0[..., 2:-2, 2:-2] - 6 * var0[..., 1:-3, 2:-2] + var0[..., :-4, 2:-2]) \
            + vm[..., 1:-1, 1:-1] * 1 / (6 * self.dy) * \
            (var0[..., 4:, 2:-2] - 6 * var0[..., 3:-1, 2:-2] + 3 * var0[..., 2:-2, 2:-2] + 2 * var0[..., 1:-3, 2:-2])

        return res

    def pv2h(self, q, hb, qb):
        """
        Potential Vorticity to SSH
        """
        qin = q[..., 1:-1, 1:-1] - qb[..., 1:-1, 1:-1]

        hrec = torch.zeros_like(q, dtype=torch.float)
        inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
        hrec[..., 1:-1, 1:-1] = inv

        hrec += hb

        return hrec

    def step(self, h0, q0, hb, qb, way=1):

        # Compute geostrophic velocities
        u, v = self.h2uv(h0)
        
        # Compute increment
        incr = self.rhs(u,v,q0,way=way)
        
        # Time integration 
        q1 = q0 + way * self.dt * incr
        
        # Elliptical inversion 
        h1 = self.pv2h(q1, hb, qb)

        return h1, q1
    
    def forward_QG(self, h0, hb=None):
        """
        Forward model time integration

        Args:
            h0: Tensor of initial SSH field with shape (N, ny, nx)
            hb: Tensor of background SSH field with shape (N, ny, nx)
            tint: Time integration length

        Returns:
            h1: Tensor of final SSH field with shape (N, ny, nx)
        """

        h0 = h0[0]

        ## unormalized the data for QG model ##
        if self.mean is not None and self.std is not None:
            h0 = h0 * self.std + self.mean
         

        if self.avg_pool is not None:
            avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
            # Apply the average pooling to each of the 15 slices along the 0th dimension
            h0_init_shape = h0.shape[1:]
            h0 = avg_pool(h0)
            
        if hb is None:
            hb = h0.clone()

        q0 = self.h2pv(h0, hb)
        qb = self.h2pv(hb, hb)

        nstep = int(self.tint / self.dt)
        h1 = h0.clone()
        q1 = q0.clone()
        for _ in range(nstep):
            h1, q1 = self.step(h1, q1, hb, qb)

        if self.avg_pool is not None:
                h1 = F.interpolate(h1.unsqueeze(0), size = h0_init_shape, mode='bilinear', align_corners=False).squeeze(0)

        ## renormalized the data for QG model ##
        if self.mean is not None and self.std is not None:
            h1 = (h1 - self.mean) / self.std 

        if self.save_debugg_path is not None:
            torch.save(h1.unsqueeze(0), self.save_debugg_path + 'qg_forwarded_state.pt')
        return h1
        
    
    def forward(self, state):
        return F.mse_loss(state[0][1:], self.forward_QG(state)[:-1])
    


# ===========================================================================
#    QG and bilin Prior Cost : phi_qg + phi_bilin
# ===========================================================================
# Description:
#    The QGCost_and_bilin cost represents a prior cost function that is
#    the sum of two prior costs : the classical trainable bilinear cost
#    and the newly introduced QG cost. The billinear cost is thus aimed
#    at learning the residu of the QG cost, allowing the 4dvarnet framework
#    to explore solutions that are a bit further away from the physics 
#    prescribed by the QG model.
#
# Formula:
#    |x - phi(x)|, where phi(x) = phi_qg(x) + phi_bilin(x)
# ===========================================================================

class QGCost_and_bilin(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad = True, domain_limits=None, res=0.05, avg_pool = None, dt=None, tint=86400, SSH=None, c=None, g=9.81, f=None, Kdiffus=None, device='cuda', save_debugg_path = None) -> None:
        super().__init__()
        print('initialisation of QG and bilinear cost')
        
        # ==============================================
        #           Initialisation of bilinear cost
        # ==============================================

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

        # ==============================================
        #           Initialisation of QG cost
        # ==============================================
        
        # average pooling before QG forward (reduce dx and dt => computation time)
        self.avg_pool = avg_pool
        if self.avg_pool is not None:
            res = res * self.avg_pool
        
        # Coordinates
        lon = torch.arange(domain_limits['lon'].start, domain_limits['lon'].stop , res, dtype=torch.float64)
        lat = torch.arange(domain_limits['lat'].start, domain_limits['lat'].stop , res, dtype=torch.float64)

        if len(lon.shape)==1:
            lon,lat = torch.meshgrid(lon,lat) 
        dx,dy = lonlat2dxdy(lon,lat)

        # Grid shapeet y
        ny, nx = dx.shape
        self.nx = nx
        self.ny = ny

        # Grid spacing
        dx = dy = (torch.nanmean(dx) + torch.nanmean(dy)) / 2
        self.dx = dx
        self.dy = dy
        
        # Time step
        self.dt = dt
        
        # Integration time (1 day)
        self.tint = tint

        # Gravity
        self.g = torch.tensor(g).to(device=device, dtype=torch.float)

        # Coriolis
        if hasattr(f, "__len__"):
            self.f = (torch.nanmean(torch.tensor(f)) * torch.ones((self.ny, self.nx)))
        elif f is not None:
            self.f = (f * torch.ones((self.ny, self.nx)))
        else:
            self.f = 4*torch.pi/86164*torch.sin(lat*torch.pi/180)
        self.f = self.f.to(device=device, dtype=torch.float)

        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = (torch.nanmean(torch.tensor(c)) * torch.ones((self.ny, self.nx))).to(device=device, dtype=torch.float)
        else:
            self.c = (c * torch.ones((self.ny, self.nx))).to(device=device, dtype=torch.float)

        # Elliptical inversion operator
        x, y = torch.meshgrid(torch.arange(1, nx - 1, dtype=torch.float),
                              torch.arange(1, ny - 1, dtype=torch.float))
        x = x.to(device=device, dtype=torch.float)
        y = y.to(device=device, dtype=torch.float)
        laplace_dst = 2 * (torch.cos(torch.pi / (nx - 1) * x) - 1) / self.dx ** 2 + \
                      2 * (torch.cos(torch.pi / (ny - 1) * y) - 1) / self.dy ** 2
        self.helmoltz_dst = self.g / self.f.mean() * laplace_dst - self.g * self.f.mean() / self.c.mean() ** 2

        # get land pixels
        if SSH is not None:
            isNAN = torch.isnan(SSH).to(device=device, dtype=torch.bool)
        else:
            isNAN = None

        ################
        # Mask array
        ################

        # mask=3 away from the coasts
        mask = torch.zeros((ny, nx), dtype=torch.int) + 3

        # mask=1 for borders of the domain 
        mask[0, :] = 1
        mask[:, 0] = 1
        mask[-1, :] = 1
        mask[:, -1] = 1

        # mask=2 for pixels adjacent to the borders 
        mask[1, 1:-1] = 2
        mask[1:-1, 1] = 2
        mask[-2, 1:-1] = 2
        mask[-3, 1:-1] = 2
        mask[1:-1, -2] = 2
        mask[1:-1, -3] = 2

        # mask=0 on land 
        if isNAN is not None:
            mask[isNAN] = 0.
            indNan = torch.argwhere(isNAN)
            for i, j in indNan:
                for p1 in range(-2, 3):
                    for p2 in range(-2, 3):
                        itest = i + p1
                        jtest = j + p2
                        if ((itest >= 0) & (itest <= ny - 1) & (jtest >= 0) & (jtest <= nx - 1)):
                            # mask=1 for coast pixels
                            if (mask[itest, jtest] >= 2) and (p1 in [-1, 0, 1] and p2 in [-1, 0, 1]):
                                mask[itest, jtest] = 1
                            # mask=1 for pixels adjacent to the coast
                            elif (mask[itest, jtest] == 3):
                                mask[itest, jtest] = 2

        self.mask = mask.to(device=device, dtype=torch.int)
        self.ind0 = (mask == 0).to(device=device, dtype=torch.bool)
        self.ind1 = (mask == 1).to(device=device, dtype=torch.bool)
        self.ind2 = (mask == 2).to(device=device, dtype=torch.bool)
        self.ind12 = (self.ind1 + self.ind2).to(device=device, dtype=torch.bool)

        # Diffusion coefficient 
        self.Kdiffus = Kdiffus

        # save debugg_path
        self.save_debugg_path = save_debugg_path

    def h2uv(self, h):
        """ SSH to U,V

        Args:
            h (2D array): SSH field.

        Returns:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
        """
    
        u = torch.zeros_like(h)
        v = torch.zeros_like(h)

        u[..., 1:-1, 1:] = - self.g / self.f[None, 1:-1, 1:] * (h[..., 2:, :-1] + h[..., 2:, 1:] - h[..., :-2, 1:] - h[..., :-2, :-1]) / (4 * self.dy)
        v[..., 1:, 1:-1] = self.g / self.f[None, 1:, 1:-1] * (h[..., 1:, 2:] + h[..., :-1, 2:] - h[..., :-1, :-2] - h[..., 1:, :-2]) / (4 * self.dx)
        
        u = torch.where(torch.isnan(u), torch.tensor(0.0), u)
        v = torch.where(torch.isnan(v), torch.tensor(0.0), v)
            
        return u, v

    def h2pv(self, h, hbc, c=None):
        """ SSH to Q

        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius

        Returns:
            q: Potential Vorticity field
        """
        
        if c is None:
            c = self.c

        q = torch.zeros_like(h, dtype=torch.float)
        q[..., 1:-1, 1:-1] = (
            self.g / self.f[None, 1:-1, 1:-1] * 
            ((h[..., 2:, 1:-1] + h[..., :-2, 1:-1] - 2 * h[..., 1:-1, 1:-1]) / self.dy ** 2 +
             (h[..., 1:-1, 2:] + h[..., 1:-1, :-2] - 2 * h[..., 1:-1, 1:-1]) / self.dx ** 2) - 
            self.g * self.f[None, 1:-1, 1:-1] / (c[None, 1:-1, 1:-1] ** 2) * h[..., 1:-1, 1:-1]
        )

        q = torch.where(torch.isnan(q), torch.tensor(0.0), q)
        q[..., self.ind12] = - self.g * self.f[None,self.ind12] / (c[None,self.ind12] ** 2) * hbc[...,self.ind12]
        q[..., self.ind0] = 0

        return q

    def rhs(self, u, v, q0, way=1):
        """ increment

        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)

        Returns:
            rhs (2D array): advection increment
        """

        # Upwind current
        u_on_T = way * 0.5 * (u[..., 1:-1, 1:-1] + u[..., 1:-1, 2:])
        v_on_T = way * 0.5 * (v[..., 1:-1, 1:-1] + v[..., 2:, 1:-1])
        up = torch.where(u_on_T < 0, torch.tensor(0.0), u_on_T)
        um = torch.where(u_on_T > 0, torch.tensor(0.0), u_on_T)
        vp = torch.where(v_on_T < 0, torch.tensor(0.0), v_on_T)
        vm = torch.where(v_on_T > 0, torch.tensor(0.0), v_on_T)

        # PV advection
        rhs_q = self._adv(up, vp, um, vm, q0)
        rhs_q[..., 2:-2, 2:-2] -= way * (self.f[None, 3:-1, 2:-2] - self.f[None, 1:-3, 2:-2]) / (2 * self.dy) * 0.5 * (v[..., 2:-2, 2:-2] + v[..., 3:-1, 2:-2])
        
        # PV Diffusion
        if self.Kdiffus is not None:
            rhs_q[..., 2:-2, 2:-2] += (
                self.Kdiffus / (self.dx ** 2) * (q0[..., 2:-2, 3:-1] + q0[..., 2:-2, 1:-3] - 2 * q0[..., 2:-2, 2:-2]) +
                self.Kdiffus / (self.dy ** 2) * (q0[..., 3:-1, 2:-2] + q0[..., 1:-3, 2:-2] - 2 * q0[..., 2:-2, 2:-2])
            )
        rhs_q = torch.where(torch.isnan(rhs_q), torch.tensor(0.0), rhs_q)
        rhs_q[..., self.ind12] = 0
        rhs_q[..., self.ind0] = 0

        return rhs_q

    def _adv(self, up, vp, um, vm, var0):
        """
            3rd-order upwind scheme.
        """

        res = torch.zeros_like(var0, dtype=torch.float)

        res[..., 2:-2,2:-2] = \
            - up[..., 1:-1, 1:-1] * 1 / (6 * self.dx) * \
            (2 * var0[..., 2:-2, 3:-1] + 3 * var0[..., 2:-2, 2:-2] - 6 * var0[..., 2:-2, 1:-3] + var0[..., 2:-2, :-4]) \
            + um[..., 1:-1, 1:-1] * 1 / (6 * self.dx) * \
            (var0[..., 2:-2, 4:] - 6 * var0[..., 2:-2, 3:-1] + 3 * var0[..., 2:-2, 2:-2] + 2 * var0[..., 2:-2, 1:-3]) \
            - vp[..., 1:-1, 1:-1] * 1 / (6 * self.dy) * \
            (2 * var0[..., 3:-1, 2:-2] + 3 * var0[..., 2:-2, 2:-2] - 6 * var0[..., 1:-3, 2:-2] + var0[..., :-4, 2:-2]) \
            + vm[..., 1:-1, 1:-1] * 1 / (6 * self.dy) * \
            (var0[..., 4:, 2:-2] - 6 * var0[..., 3:-1, 2:-2] + 3 * var0[..., 2:-2, 2:-2] + 2 * var0[..., 1:-3, 2:-2])

        return res

    def pv2h(self, q, hb, qb):
        """
        Potential Vorticity to SSH
        """
        qin = q[..., 1:-1, 1:-1] - qb[..., 1:-1, 1:-1]

        hrec = torch.zeros_like(q, dtype=torch.float)
        inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
        hrec[..., 1:-1, 1:-1] = inv

        hrec += hb

        return hrec

    def step(self, h0, q0, hb, qb, way=1):
  
        # Compute geostrophic velocities
        u, v = self.h2uv(h0)
        
        # Compute increment
        incr = self.rhs(u,v,q0,way=way)
        
        # Time integration 
        q1 = q0 + way * self.dt * incr
        
        # Elliptical inversion 
        h1 = self.pv2h(q1, hb, qb)

        return h1, q1

    def forward_QG(self, h0, save = True):
        """
        Forward QG model integration

        Args:
            h0: Tensor of initial SSH field with shape (B, D, ny, nx) with B=1 (batch size =1), D = 15 typically
        Returns:
            h1: Tensor of final SSH given by D independant QG integrations of 24 hours, with shape (D, ny, nx).
        """
        with torch.no_grad():
            h0 = h0[0]
            h0 = torch.nan_to_num(h0, nan=0.0)

            # Define an average pooling layer with a kernel size of 2x2 and a stride of 2
            if self.avg_pool is not None:
                avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                # Apply the average pooling to each of the 15 slices along the 0th dimension
                h0_init_shape = h0.shape[1:]
                h0 = avg_pool(h0)

            # hb: Tensor of background SSH field with shape (N, ny, nx)
            hb = h0.clone()
            q0 = self.h2pv(h0, hb)
            qb = self.h2pv(hb, hb)
            nstep = int(self.tint / self.dt)
            h1 = h0.clone()
            q1 = q0.clone()
            if save:
                ssh_hourly_forecasts = [] 
                ns_hourly = [int((h* 3600)/ self.dt)+1 for h in range(round(self.tint/3600))]
            for _ in range(nstep):
                h1, q1 = self.step(h1, q1, hb, qb)
                if save and _ in ns_hourly:
                    ssh_hourly_forecasts.append(h1)

            if self.save_debugg_path is not None:
                torch.save(h1.unsqueeze(0), self.save_debugg_path + 'qg_forwarded_state.pt')
            if torch.isnan(h1).any():
                raise ValueError("NaN detected in h1, saving last clean state, and forwarded state")
            
            if self.avg_pool is not None:
                h1 = F.interpolate(h1.unsqueeze(0), size = h0_init_shape, mode='bilinear', align_corners=False).squeeze(0)
            return h1
        
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
        try:
            return F.mse_loss(state[0][1:], self.forward_QG(state)[:-1] + self.forward_ae(state).squeeze(0)[:-1])
        except:
            print('self.forward_QG(state)[:-1].shape : ',self.forward_QG(state)[:-1].shape)
            print('self.forward_ae(state)[:].shape : ',self.forward_ae(state)[:].shape)
            raise ValueError
            
    

# ==============================================================================================
#    QG weak fourdvar Cost : prior cost integrating the QG model under weak 4dvar formulation
# ==============================================================================================
# Description:
#    The QG weak fourdvar Cost represents a prior cost that integrates
#    the Quasi-Geostrophic (QG) model, but relates to weak 4dvar formulation.
#    We proceed to only one forward integration of QG model over N = 15 days, given the
#    state at day 0. 
#
# Formula:
#    |x - phi_qg_weak(x)|
#
# Details:
#    - phi_qg_weak(x) corresponds to one forward integration
#      of the QG model over a time span of N = 15 days (batch lenght).
#    - The integration has a time step typically < 10 minutes.
#    - Specifically:
#        phi_qg_weak(x)(t) = forward_qg(x(0),t) with forward_qg(x(0),t) beeing the forward integration 
#        of QG over t days starting from state at day 0.
#        ie phi_qg_weak : x -> [forward_qg(x(0),0),forward_qg(x(0),1), .. , forward_qg(x(0),N-1)]
# ================================================================================================


class QGCost_weak_fourdvar(nn.Module): 
    def __init__(self, domain_limits=None, res=0.05, dt=None, nb_days_int=None, SSH=None, c=None, g=9.81, f=None, Kdiffus=None, device='cuda', save_debugg_path = None) -> None:
        super().__init__()
        print('initialisation of QG')
        

        # Integration time (1 day)
        self.tint = nb_days_int * 86400

        # Coordinates
        lon = torch.arange(domain_limits['lon'].start, domain_limits['lon'].stop , res, dtype=torch.float64)
        lat = torch.arange(domain_limits['lat'].start, domain_limits['lat'].stop , res, dtype=torch.float64)
        #print('lon ', lon)
        if len(lon.shape)==1:
            lon,lat = torch.meshgrid(lon,lat)
        dx,dy = lonlat2dxdy(lon,lat)

        # Grid shape
        ny, nx = dx.shape
        self.nx = nx
        self.ny = ny

        # Grid spacing
        dx = dy = (torch.nanmean(dx) + torch.nanmean(dy)) / 2
        self.dx = dx
        self.dy = dy
        # Time step
        self.dt = dt

        # Gravity
        self.g = torch.tensor(g).to(device=device, dtype=torch.float)
        
        # Coriolis
        if hasattr(f, "__len__"):
            self.f = (torch.nanmean(torch.tensor(f)) * torch.ones((self.ny, self.nx)))
        elif f is not None:
            self.f = (f * torch.ones((self.ny, self.nx)))
        else:
            self.f = 4*torch.pi/86164*torch.sin(lat*torch.pi/180)
        self.f = self.f.to(device=device, dtype=torch.float)

        # Rossby radius
        if hasattr(c, "__len__"):
            self.c = (torch.nanmean(torch.tensor(c)) * torch.ones((self.ny, self.nx))).to(device=device, dtype=torch.float)
        else:
            self.c = (c * torch.ones((self.ny, self.nx))).to(device=device, dtype=torch.float)

        # Elliptical inversion operator
        x, y = torch.meshgrid(torch.arange(1, nx - 1, dtype=torch.float),
                              torch.arange(1, ny - 1, dtype=torch.float))
        x = x.to(device=device, dtype=torch.float)
        y = y.to(device=device, dtype=torch.float)
        laplace_dst = 2 * (torch.cos(torch.pi / (nx - 1) * x) - 1) / self.dx ** 2 + \
                      2 * (torch.cos(torch.pi / (ny - 1) * y) - 1) / self.dy ** 2
        self.helmoltz_dst = self.g / self.f.mean() * laplace_dst - self.g * self.f.mean() / self.c.mean() ** 2

        # get land pixels
        if SSH is not None:
            isNAN = torch.isnan(SSH).to(device=device, dtype=torch.bool)
        else:
            isNAN = None

        ################
        # Mask array
        ################

        # mask=3 away from the coasts
        mask = torch.zeros((ny, nx), dtype=torch.int) + 3

        # mask=1 for borders of the domain 
        mask[0, :] = 1
        mask[:, 0] = 1
        mask[-1, :] = 1
        mask[:, -1] = 1

        # mask=2 for pixels adjacent to the borders 
        mask[1, 1:-1] = 2
        mask[1:-1, 1] = 2
        mask[-2, 1:-1] = 2
        mask[-3, 1:-1] = 2
        mask[1:-1, -2] = 2
        mask[1:-1, -3] = 2

        # mask=0 on land 
        if isNAN is not None:
            mask[isNAN] = 0.
            indNan = torch.argwhere(isNAN)
            for i, j in indNan:
                for p1 in range(-2, 3):
                    for p2 in range(-2, 3):
                        itest = i + p1
                        jtest = j + p2
                        if ((itest >= 0) & (itest <= ny - 1) & (jtest >= 0) & (jtest <= nx - 1)):
                            # mask=1 for coast pixels
                            if (mask[itest, jtest] >= 2) and (p1 in [-1, 0, 1] and p2 in [-1, 0, 1]):
                                mask[itest, jtest] = 1
                            # mask=1 for pixels adjacent to the coast
                            elif (mask[itest, jtest] == 3):
                                mask[itest, jtest] = 2

        self.mask = mask.to(device=device, dtype=torch.int)
        self.ind0 = (mask == 0).to(device=device, dtype=torch.bool)
        self.ind1 = (mask == 1).to(device=device, dtype=torch.bool)
        self.ind2 = (mask == 2).to(device=device, dtype=torch.bool)
        self.ind12 = (self.ind1 + self.ind2).to(device=device, dtype=torch.bool)

        # Diffusion coefficient 
        self.Kdiffus = Kdiffus

    def h2uv(self, h):
        """ SSH to U,V

        Args:
            h (2D array): SSH field.

        Returns:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
        """
    
        u = torch.zeros_like(h)
        v = torch.zeros_like(h)

        u[..., 1:-1, 1:] = - self.g / self.f[None, 1:-1, 1:] * (h[..., 2:, :-1] + h[..., 2:, 1:] - h[..., :-2, 1:] - h[..., :-2, :-1]) / (4 * self.dy)
        v[..., 1:, 1:-1] = self.g / self.f[None, 1:, 1:-1] * (h[..., 1:, 2:] + h[..., :-1, 2:] - h[..., :-1, :-2] - h[..., 1:, :-2]) / (4 * self.dx)
        
        u = torch.where(torch.isnan(u), torch.tensor(0.0), u)
        v = torch.where(torch.isnan(v), torch.tensor(0.0), v)
            
        return u, v

    def h2pv(self, h, hbc, c=None):
        """ SSH to Q

        Args:
            h (2D array): SSH field.
            c (2D array): Phase speed of first baroclinic radius

        Returns:
            q: Potential Vorticity field
        """
        
        if c is None:
            c = self.c

        q = torch.zeros_like(h, dtype=torch.float)
        q[..., 1:-1, 1:-1] = (
            self.g / self.f[None, 1:-1, 1:-1] * 
            ((h[..., 2:, 1:-1] + h[..., :-2, 1:-1] - 2 * h[..., 1:-1, 1:-1]) / self.dy ** 2 +
             (h[..., 1:-1, 2:] + h[..., 1:-1, :-2] - 2 * h[..., 1:-1, 1:-1]) / self.dx ** 2) - 
            self.g * self.f[None, 1:-1, 1:-1] / (c[None, 1:-1, 1:-1] ** 2) * h[..., 1:-1, 1:-1]
        )

        q = torch.where(torch.isnan(q), torch.tensor(0.0), q)
        q[..., self.ind12] = - self.g * self.f[None,self.ind12] / (c[None,self.ind12] ** 2) * hbc[...,self.ind12]
        q[..., self.ind0] = 0

        return q

    def rhs(self, u, v, q0, way=1):
        """ increment

        Args:
            u (2D array): Zonal velocity
            v (2D array): Meridional velocity
            q : PV start
            way: forward (+1) or backward (-1)

        Returns:
            rhs (2D array): advection increment
        """

        # Upwind current
        u_on_T = way * 0.5 * (u[..., 1:-1, 1:-1] + u[..., 1:-1, 2:])
        v_on_T = way * 0.5 * (v[..., 1:-1, 1:-1] + v[..., 2:, 1:-1])
        up = torch.where(u_on_T < 0, torch.tensor(0.0), u_on_T)
        um = torch.where(u_on_T > 0, torch.tensor(0.0), u_on_T)
        vp = torch.where(v_on_T < 0, torch.tensor(0.0), v_on_T)
        vm = torch.where(v_on_T > 0, torch.tensor(0.0), v_on_T)

        # PV advection
        rhs_q = self._adv(up, vp, um, vm, q0)
        rhs_q[..., 2:-2, 2:-2] -= way * (self.f[None, 3:-1, 2:-2] - self.f[None, 1:-3, 2:-2]) / (2 * self.dy) * 0.5 * (v[..., 2:-2, 2:-2] + v[..., 3:-1, 2:-2])
        
        # PV Diffusion
        if self.Kdiffus is not None:
            rhs_q[..., 2:-2, 2:-2] += (
                self.Kdiffus / (self.dx ** 2) * (q0[..., 2:-2, 3:-1] + q0[..., 2:-2, 1:-3] - 2 * q0[..., 2:-2, 2:-2]) +
                self.Kdiffus / (self.dy ** 2) * (q0[..., 3:-1, 2:-2] + q0[..., 1:-3, 2:-2] - 2 * q0[..., 2:-2, 2:-2])
            )
        rhs_q = torch.where(torch.isnan(rhs_q), torch.tensor(0.0), rhs_q)
        rhs_q[..., self.ind12] = 0
        rhs_q[..., self.ind0] = 0

        return rhs_q

    def _adv(self, up, vp, um, vm, var0):
        """
            3rd-order upwind scheme.
        """

        res = torch.zeros_like(var0, dtype=torch.float)

        res[..., 2:-2,2:-2] = \
            - up[..., 1:-1, 1:-1] * 1 / (6 * self.dx) * \
            (2 * var0[..., 2:-2, 3:-1] + 3 * var0[..., 2:-2, 2:-2] - 6 * var0[..., 2:-2, 1:-3] + var0[..., 2:-2, :-4]) \
            + um[..., 1:-1, 1:-1] * 1 / (6 * self.dx) * \
            (var0[..., 2:-2, 4:] - 6 * var0[..., 2:-2, 3:-1] + 3 * var0[..., 2:-2, 2:-2] + 2 * var0[..., 2:-2, 1:-3]) \
            - vp[..., 1:-1, 1:-1] * 1 / (6 * self.dy) * \
            (2 * var0[..., 3:-1, 2:-2] + 3 * var0[..., 2:-2, 2:-2] - 6 * var0[..., 1:-3, 2:-2] + var0[..., :-4, 2:-2]) \
            + vm[..., 1:-1, 1:-1] * 1 / (6 * self.dy) * \
            (var0[..., 4:, 2:-2] - 6 * var0[..., 3:-1, 2:-2] + 3 * var0[..., 2:-2, 2:-2] + 2 * var0[..., 1:-3, 2:-2])

        return res

    def pv2h(self, q, hb, qb):
        """
        Potential Vorticity to SSH
        """
        qin = q[..., 1:-1, 1:-1] - qb[..., 1:-1, 1:-1]

        hrec = torch.zeros_like(q, dtype=torch.float)
        inv = inverse_elliptic_dst(qin, self.helmoltz_dst)
        hrec[..., 1:-1, 1:-1] = inv

        hrec += hb

        return hrec

    def step(self, h0, q0, hb, qb, way=1):
  
        # Compute geostrophic velocities
        u, v = self.h2uv(h0)
        
        # Compute increment
        incr = self.rhs(u,v,q0,way=way)
        
        # Time integration 
        q1 = q0 + way * self.dt * incr
        
        # Elliptical inversion 
        h1 = self.pv2h(q1, hb, qb)

        return h1, q1

    def forward_QG(self, h0, save = False):
        """
        Forward QG model integration

        Args:
            h0: Tensor of initial SSH field with shape (B, D, ny, nx) with B=1 (batch size =1), D = 15 typically
        Returns:
            h1: Tensor of final SSH given by one single QG integration over D days initialized with SSH at D = 0, with shape (D, ny, nx).
        """
        with torch.no_grad():
            h0 = h0[0][0].unsqueeze(0)
            h0 = torch.nan_to_num(h0, nan=0.0)

            # hb: Tensor of background SSH field with shape (N, ny, nx)
            hb = h0.clone()
            q0 = self.h2pv(h0, hb)
            qb = self.h2pv(hb, hb)
            nstep = int(self.tint / self.dt)
            h1 = h0.clone()
            q1 = q0.clone()
            
            ssh_daily_forecasts = []
            ns_daily= [int((h* 86400)/ self.dt) for h in range(round(self.tint/86400))]
            for _ in range(nstep):
                h1, q1 = self.step(h1, q1, hb, qb)
                if _ in ns_daily:
                    ssh_daily_forecasts.append(h1[0])
            stacked_ssh = torch.stack(ssh_daily_forecasts, dim=0)
            
            if self.save_debugg_path is not None:
                torch.save(stacked_ssh.unsqueeze(0), self.save_debugg_path + 'qg_forwarded_state.pt')
            return stacked_ssh

    def forward(self, state):
        return F.mse_loss(state[0][1:], self.forward_QG(state)[1:])
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from contrib.stoch_vae.VAE import VAE as VAE_attention

class VAE(nn.Module):
    def __init__(self, input_shape, z_dim):
        super().__init__()

        self.dim_in = input_shape[0]
        self.z_dim = z_dim
        self.input_shape = input_shape

        # encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.dim_in, 30, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(60, 120, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_out_size = self._get_conv_out_size(input_shape)

        self.mu = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.log_var = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        # decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(z_dim, self.conv_out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(120, 60, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(60, 30, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(30, self.dim_in, 3, stride=2, padding=1, output_padding=1)
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size()[0], -1)
        mu_p = self.mu(x)
        log_var_p = self.log_var(x)
        return mu_p, log_var_p

    def reparametrization(self, mu, log_var):
        epsilon = torch.Tensor(np.random.normal(size=(self.z_dim), scale=1.0))
        epsilon = epsilon.to(mu.device)
        return mu + epsilon * torch.exp(log_var / 2)

    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.view(x.size()[0], *self.conv_out_shape[1:])
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu_p, log_var_p = self.encode(x)
        x = self.reparametrization(mu_p, log_var_p)
        images_p = self.decode(x)
        return mu_p, log_var_p, images_p

    def project_latent_space(self,x):
        mu_p, log_var_p = self.encode(x)
        x = self.reparametrization(mu_p, log_var_p)
        return x

    def vae_loss(self, x, x_hat, mean, log_var, wKL=.01):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + wKL*KLD, reproduction_loss, KLD

    def _get_conv_out_size(self, shape):
        out = self.encoder_conv(torch.zeros(1, *shape))
        self.conv_out_shape = out.size()
        return int(np.prod(self.conv_out_shape))

class VAE2(nn.Module):
    def __init__(self, input_shape, z_dim):
        super().__init__()

        self.dim_in = input_shape[0]
        self.z_dim = z_dim
        self.input_shape = input_shape

        # encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.dim_in, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.conv_out_size = self._get_conv_out_size(input_shape)

        self.mu = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.log_var = nn.Sequential(
            nn.Linear(self.conv_out_size, z_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        # decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(z_dim, self.conv_out_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.decoder_conv = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(32, self.dim_in, 3, stride=1, padding=1),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size()[0], -1)
        mu_p = self.mu(x)
        log_var_p = self.log_var(x)
        return mu_p, log_var_p

    def reparametrization(self, mu, log_var):
        epsilon = torch.Tensor(np.random.normal(size=(self.z_dim), scale=1.0))
        epsilon = epsilon.to(mu.device)
        return mu + epsilon * torch.exp(log_var / 2)

    def decode(self, x):
        x = self.decoder_linear(x)
        x = x.view(x.size()[0], *self.conv_out_shape[1:])
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu_p, log_var_p = self.encode(x)
        x = self.reparametrization(mu_p, log_var_p)
        images_p = self.decode(x)
        return mu_p, log_var_p, images_p

    def project_latent_space(self,x):
        mu_p, log_var_p = self.encode(x)
        x = self.reparametrization(mu_p, log_var_p)
        return x

    def vae_loss(self, x, x_hat, mean, log_var, wKL=.01):
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + wKL*KLD, reproduction_loss, KLD

    def _get_conv_out_size(self, shape):
        out = self.encoder_conv(torch.zeros(1, *shape))
        self.conv_out_shape = out.size()
        return int(np.prod(self.conv_out_shape))

class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, gen_mod, grad_mod, n_step, lr_grad=0.2, **kwargs):
        super().__init__()

        # Need 4 models (prior, obs, gen(x), solver-J) )
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.gen_mod = gen_mod
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.lr_grad = lr_grad

        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init

        return batch.input.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, step):

        # assimilation in latent space
        z_state = self.gen_mod.project_latent_space(state)
        var_cost = self.prior_cost(state) + self.obs_cost(self.gen_mod.decoder(z_state), batch)
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

                # plot
                '''
                if step>=0:              
                    data = xr.Dataset(data_vars=dict(
                               out=(["time", "lat", "lon"], state[0].detach().cpu().data),
                               ),
                          coords=(dict(time=range(15),
                                  lon=range(120),
                                  lat=range(120))))
                    data.out.plot(col='time',col_wrap=5)
                    plt.show()
                '''

                if not self.training:
                    state = state.detach().requires_grad_(True)

            #if not self.training:
            #    state = self.prior_cost.forward_ae(state)
            #state = self.prior_cost.forward_ae(state)
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
    def __init__(self, dim_in, dim_hidden, gen_mod, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):
        super().__init__()
        self.nt = nt
        self.gen_mod = gen_mod
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
        # done in latent space
        z_state = self.gen_mod.project_latent_space(state)
        phi_z_state = self.gen_mod.project_latent_space(self.forward_ae(state))
        return F.mse_loss(z_state, phi_z_state)

class RearrangedBilinAEPriorCost(BilinAEPriorCost):
    """
    Wrapper around the base prior cost that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, rearrange_from='b c t', rearrange_to='b t c ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

    def forward_ae(self, x):
        x = einops.rearrange(x, self.rearrange_bef)
        x = super().forward_ae(x)
        x = einops.rearrange(x, self.rearrange_aft)
        return x


class RearrangedConvLstmGradModel(ConvLstmGradModel):
    """
    Wrapper around the base grad model that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, rearrange_from='b c t', rearrange_to='b t c ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        super().reset_state(inp)

    def forward(self, x):
        x = einops.rearrange(x, self.rearrange_bef)
        x = super().forward(x)
        x = einops.rearrange(x, self.rearrange_aft)
        return x

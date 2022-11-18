import numpy as np
import pytorch_lightning as pl
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools as it
import xarray as xr

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight):
        super().__init__()
        self.solver = solver
        self.rec_weight = nn.Parameter(torch.from_numpy(rec_weight), requires_grad=False)
        self.test_data = None

    @staticmethod
    def weighted_mse(err, weight):
        err_w = (err * weight[None, ...])
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000., device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def training_step(self, batch, batch_idx):
        loss, grad_loss = self.step(batch, 'tr', training=True)[0]
        return loss + 50*grad_loss

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')[0]

    def forward(self, batch, training=False):
        return self.solver(batch)

    def step(self, batch, phase='', opt_idx=None, training=False):
        states = self(batch=batch, training=training)
        loss = sum(
                self.weighted_mse(state - batch.tgt, self.rec_weight)
            for state in states)

        grad_loss = sum(
                self.weighted_mse(
                    kornia.filters.sobel(state) - kornia.filters.sobel(batch.tgt),
                    self.rec_weight
                ) for state in states)
        out = states[-1]
        rmse = self.weighted_mse(out - batch.tgt, self.rec_weight)**0.5
        self.log(f'{phase}_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{phase}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f'{phase}_gloss', grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        return [loss, grad_loss], out

    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.solver.grad_mod.parameters(), 'lr':1e-3},
            {'params': self.solver.prior_cost.parameters(), 'lr':5e-4}],
            lr=1e-3)

    def test_step(self, batch, batch_idx):
        out = self(batch=batch)[-1]

        return torch.stack([
            batch.tgt.cpu(),
            out.squeeze(dim=-1).detach().cpu(),
            ],dim=1)

    def test_epoch_end(self, outputs):
        rec_data = (it.chain(lt) for lt in zip(*outputs))
        rec_da = (
            self.trainer
            .test_dataloaders[0].dataset
            .reconstruct(rec_data, self.rec_weight.cpu().numpy())
        )
        npa = rec_da.values
        lonidx = ~np.all(np.isnan(npa), axis=tuple([0, 1, 2]))
        latidx = ~np.all(np.isnan(npa), axis=tuple([0, 1, 3]))
        tidx = ~np.all(np.isnan(npa), axis=tuple([0, 2, 3]))

        self.test_data = xr.Dataset({
            k: rec_da.isel(v0=i,
                           time=tidx, lat=latidx, lon=lonidx
                        )
            for i, k  in enumerate(['ssh', 'rec_ssh'])
        })

class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, cut_graph_freq):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.cut_graph_freq = cut_graph_freq

        self._grad_norm = None

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        if self._grad_norm is None:
            self._grad_norm = (grad**2).mean().sqrt()
        
        state_update = 1 / (step + 1)  *  self.grad_mod(grad / self._grad_norm)
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            _intermediate_states = []
            state = batch.input.nan_to_num().detach().requires_grad_(True)
            self.grad_mod.reset_state(batch.input)
            self._grad_norm = None

            for step in range(self.n_step):

                if step + 1 % self.cut_graph_freq == 0:
                    _intermediate_states.append(state)
                    state = state.detach().requires_grad_(True)
                    self.grad_mod.detach_state()

                state = self.solver_step(state, batch, step=step)

        return [*_intermediate_states, state]



class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.25):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._state = [
                torch.zeros(size, device=inp.device),
                torch.zeros(size, device=inp.device),
        ]

    def detach_state(self):
        self._state = [
                s.detach().requires_grad_(True) for s in self._state
        ]

    def forward(self, x):
        hidden, cell = self._state
        x = self.dropout(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(torch.sigmoid, [in_gate, remember_gate, out_gate])
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        hidden = self.dropout(hidden)
        return self.conv_out(hidden)


class BaseObsCost(nn.Module):
    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3):
        super().__init__()
        self.conv_in = nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv_hidden = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)

        self.bilin_1 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bilin_21 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bilin_22 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
    
        self.conv_out = nn.Conv2d(2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size//2)


    def forward_ae(self, x):
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        return self.conv_out(
            torch.cat([self.bilin_1(x), self.bilin_21(x) * self.bilin_21(x)], dim=1)
        )

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state))


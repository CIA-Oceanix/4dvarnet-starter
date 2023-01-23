import numpy as np
import pytorch_lightning as pl
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, opt_fn=None, norm_stats=None):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight))
        self.test_data = None
        self.norm_stats = norm_stats if norm_stats is not None else (0., 1.)
        self.opt_fn = opt_fn

    @staticmethod
    def weighted_mse(err, weight):
        err_w = (err * weight[None, ...])
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000., device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def training_epoch_end(self, outputs):
        best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        losses = torch.stack([out['loss'] for out in outputs])

        can_reload_ckpt = len(best_ckpt_path) > 0

        should_reload_ckpt_1 = not losses.isfinite().all()
        should_reload_ckpt_2 = (losses.argmax() > losses.argmin()) and (losses.max() > (10 * losses.min()))
        should_reload_ckpt = should_reload_ckpt_1 or should_reload_ckpt_2

        if should_reload_ckpt and can_reload_ckpt:
            self.load_state_dict(torch.load(best_ckpt_path)['state_dict'])

    def training_step(self, batch, batch_idx):
        loss, grad_loss, prior_cost = self.step(batch, 'tr')[0]
        return 50*loss + 1000*grad_loss + 0.5*prior_cost

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')[0]

    def forward(self, batch):
        return self.solver(batch)

    def step(self, batch, phase='', opt_idx=None):
        states = self(batch=batch)
        loss = sum(
                (i+1) * self.weighted_mse(state - batch.tgt, self.rec_weight)
            for i,state in enumerate(states))

        out = states[-1]
        grad_loss = self.weighted_mse(
            kornia.filters.sobel(out) - kornia.filters.sobel(batch.tgt),
            self.rec_weight
        )
        prior_cost = self.solver.prior_cost(out)
        with torch.no_grad():
            rmse = self.weighted_mse((out - batch.tgt) * self.norm_stats[1], self.rec_weight)**0.5
            self.log(f'{phase}_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{phase}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{phase}_gloss', grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        return [loss, grad_loss, prior_cost], out

    def configure_optimizers(self):
        if self.opt_fn is None:
            return torch.optim.Adam(
                [{'params': self.solver.grad_mod.parameters(), 'lr':1e-3},
                {'params': self.solver.prior_cost.parameters(), 'lr':5e-4}],
            )
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        out = self(batch=batch)[-1]
        m, s = self.norm_stats

        return torch.stack([
            batch.tgt.cpu() * s + m,
            out.squeeze(dim=-1).detach().cpu() * s + m,
        ], dim=1)

    def test_epoch_end(self, outputs):
        rec_data = outputs
        rec_da = (
            self.trainer.test_dataloaders[0].dataset
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
    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, cut_graph_freq, lr_grad=0.1):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod

        self.n_step = n_step
        self.cut_graph_freq = cut_graph_freq
        self.lr_grad = lr_grad

        self._grad_norm = None


    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        if self._grad_norm is None:
            self._grad_norm = (grad**2).mean().sqrt()
        
        state_update = (
            1 / (step + 1)  *  self.grad_mod(grad / self._grad_norm) 
                + self.lr_grad * (step + 1) / self.n_step * grad
        )
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
                if not self.training:
                    state = state.detach().requires_grad_(True)
        

        if not self.training:
            state = self.prior_cost.forward_ae(state)
        output = [*_intermediate_states, state]

        return output

class IdentityGradModel(nn.Module):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def reset_state(self, *args, **kwargs):
        pass

    def detach_state(self):
        pass

    def forward(self, x):
        return self.lr * x


class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
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

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = nn.UpsamplingBilinear2d(scale_factor=downsamp) if downsamp is not None else nn.Identity()

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._state = [
                self.down(torch.zeros(size, device=inp.device)),
                self.down(torch.zeros(size, device=inp.device)),
        ]

    def detach_state(self):
        self._state = [
                s.detach().requires_grad_(True) for s in self._state
        ]

    def forward(self, x):
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(torch.sigmoid, [in_gate, remember_gate, out_gate])
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        return out


class BaseObsCost(nn.Module):
    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None):
        super().__init__()
        self.conv_in = nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv_hidden = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)

        self.bilin_1 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bilin_21 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bilin_22 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
    
        self.conv_out = nn.Conv2d(2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size//2)

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = nn.UpsamplingBilinear2d(scale_factor=downsamp) if downsamp is not None else nn.Identity()

    def forward_ae(self, x):
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        x = self.conv_out(
            torch.cat([self.bilin_1(x),
                       self.bilin_21(x) * self.bilin_21(x)], dim=1)
        )
        x = self.up(x)
        return  x

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state))


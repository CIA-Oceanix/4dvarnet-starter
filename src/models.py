import numpy as np
import pytorch_lightning as pl
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def training_epoch_end(self, outputs):
        best_ckpt_path = self.trainer.checkpoint_callback.best_model_path
        if len(best_ckpt_path) > 0:
            def should_reload_ckpt(losses):
                if losses.argmax() < losses.argmin():
                    return False
                if losses.max() > (10 * losses.min()):
                    print("Reloading because of check", 1)
                    return True

            if should_reload_ckpt(torch.stack([out['loss'] for out in outputs])):
                print('reloading', best_ckpt_path)
                ckpt = torch.load(best_ckpt_path)
                self.load_state_dict(ckpt['state_dict'])

    def training_step(self, batch, batch_idx):
        loss, grad_loss, prior_cost = self.step(batch, 'tr', training=True)[0]
        return 50*loss + 1000*grad_loss + 0.5*prior_cost

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')[0]

    def forward(self, batch, training=False):
        return self.solver(batch, training=training)

    def step(self, batch, phase='', opt_idx=None, training=False):
        states = self(batch=batch, training=training)
        loss = sum(
                # (len(state) - i)/len(state) * self.weighted_mse(state - batch.tgt, self.rec_weight)
                # (i+1)/len(state) * self.weighted_mse(state - batch.tgt, self.rec_weight)
                (i+1) * self.weighted_mse(state - batch.tgt, self.rec_weight)
            for i,state in enumerate(states))

        out = states[-1]
        grad_loss = self.weighted_mse(
            kornia.filters.sobel(out) - kornia.filters.sobel(batch.tgt),
            self.rec_weight
        )
        prior_cost = self.solver.prior_cost(out)
        with torch.no_grad():
            rmse = self.weighted_mse(out - batch.tgt, self.rec_weight)**0.5
            self.log(f'{phase}_rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{phase}_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'{phase}_gloss', grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        return [loss, grad_loss, prior_cost], out

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            [{'params': self.solver.grad_mod.parameters(), 'lr':1e-3},
            {'params': self.solver.prior_cost.parameters(), 'lr':5e-4}],
        )
        return opt

    def test_step(self, batch, batch_idx):
        out = self(batch=batch)[-1]

        return torch.stack([
            batch.tgt.cpu(),
            out.squeeze(dim=-1).detach().cpu(),
        ], dim=1)

    def test_epoch_end(self, outputs):
        rec_data = outputs
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
        self.post_conv = grad_mod

        self.n_step = n_step
        self.cut_graph_freq = cut_graph_freq

        self._grad_norm = None

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]

        if self._grad_norm is None:
            self._grad_norm = (grad**2).mean().sqrt()
        
        state_update = (
            1 / (step + 1)  *  self.grad_mod(grad / self._grad_norm) 
                +  0.1 * (step + 1) / self.n_step 
        )
        return state - state_update

    def forward(self, batch, training=False):
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

        output = [*_intermediate_states, state]

        if not training:
            return [t.detach() for t in output]

        return output



class ConvLstmGradModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1):
        super().__init__()
        self.down = nn.AvgPool2d(2)
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
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self._state = []

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._state = [
                torch.zeros(size, device=inp.device),
                torch.zeros(size, device=inp.device),
        ]
        # self._state = [
        #         self.down(torch.zeros(size, device=inp.device)),
        #         self.down(torch.zeros(size, device=inp.device)),
        # ]

    def detach_state(self):
        self._state = [
                s.detach().requires_grad_(True) for s in self._state
        ]

    def forward(self, x):
        hidden, cell = self._state
        x = self.dropout(x)
        # x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(torch.sigmoid, [in_gate, remember_gate, out_gate])
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        # hidden = self.up(hidden)
        out = self.conv_out(hidden)
        return out


class BaseObsCost(nn.Module):
    def forward(self, state, batch):
        msk = batch.input.isfinite()
        return F.mse_loss(state[msk], batch.input.nan_to_num()[msk])


class BilinAEPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3):
        super().__init__()
        self.conv_in = nn.Conv2d(dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.down = nn.AvgPool2d(2)
        self.conv_hidden = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)

        self.bilin_1 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bilin_21 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
        self.bilin_22 = nn.Conv2d(dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size//2)
    
        self.conv_out = nn.Conv2d(2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size//2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward_ae(self, x):
        # x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        x = self.conv_out(
            torch.cat([self.bilin_1(x),
                       self.bilin_21(x) * self.bilin_21(x)], dim=1)
        )
        # x = self.up(x)
        return  x

    def forward(self, state):
        return F.mse_loss(state, self.forward_ae(state))


import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from .models_spde import *
import cupy
from cupyx.scipy.sparse.linalg import spsolve as cupy_spsolve
from cupyx.scipy.sparse import csc_matrix as cupy_sp_csc_matrix
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, n_simu=2):
        super().__init__()
        self.solver = solver
        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.n_simu = n_simu

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

    def run_simulation(self,i,tau,M,x,dx,dy,dt,n_init_run=10):
        nb_nodes = M.shape[0]
        regul = (tau*np.sqrt(dt))/(dx*dy)
        val = cupy.fromDlpack(to_dlpack(M.coalesce().values().data))
        ind = cupy.fromDlpack(to_dlpack(M.coalesce().indices().data))
        M_ = cupy_sp_csc_matrix((val,ind),shape=(M.size()[0],M.size()[0]))
        # if i==0: start stability run
        if i==0:
            xi = torch.randn(nb_nodes).to(device)
            for i in range(n_init_run):
                random = torch.randn(nb_nodes).to(device)
                RM = torch.mul(regul,random)+torch.flatten(xi)
                RM_ = cupy.fromDlpack(to_dlpack(RM))
                xi_ = cupy_spsolve(M_, RM_)
                xi = torch.flatten(from_dlpack(xi_.toDlpack())).to(device)
                #xi = torch.flatten(cupy_solve_sparse.apply(M,RM)).to(device)     
        else:
            random = torch.randn(nb_nodes).to(device)
            RM = torch.mul(regul,random)+torch.flatten(x[i-1])
            RM_ = cupy.fromDlpack(to_dlpack(RM))
            xi_ = cupy_spsolve(M_, RM_)
            xi = torch.flatten(from_dlpack(xi_.toDlpack())).to(device)
            #xi = torch.flatten(cupy_solve_sparse.apply(M,RM)).to(device)
        xi.requires_grad = True
        xi = torch.flatten(xi)
        x.append(xi)
        return x

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def forward(self, batch):
        return self.solver(batch)

    def step(self, batch, phase=""):
        if self.training and batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        loss, out, theta = self.base_step(batch, phase)
        # MSE grad loss
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.rec_weight)
        self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
        _, theta_nrs = self.solver(batch, reshape_theta = False)
        prior_cost = self.solver.prior_cost(self.solver.init_state(batch, torch.cat((out, theta_nrs), dim=1)),
                                            exclude_params=True)
        training_loss = 50 * loss  + 1000 * grad_loss + 1.0 * prior_cost

        if self.current_epoch>=87:
            if self.solver.aug_state==True:
                nll_loss = torch.nanmean(self.solver.nll(batch.tgt,
                                                 theta = theta, 
                                                 #mu = out,
                                                 det=True))
            else:
                nll_loss = torch.nanmean(self.solver.nll(batch.tgt,
                                                 theta = self.solver.nll.encoder(out),
                                                 det=True))
            self.log( f"{phase}_nll_loss", nll_loss, prog_bar=True, on_step=False, on_epoch=True)
            training_loss += nll_loss * 1e-6
            print(50 * loss  + 1000 * grad_loss + 1.0 * prior_cost, nll_loss*1e-6)
        return training_loss, out

    def base_step(self, batch, phase=""):
        out, theta = self(batch=batch)

        # mse loss
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out, theta

    def configure_optimizers(self):
        return self.opt_fn(self)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
            self.test_params = []

        # get norm_stats
        m, s = self.norm_stats

        for i in range(self.n_simu):
            self.test_data.append([])

            # 4DVarNets scheme
            out, theta = self(batch=batch)

            n_b, n_t, n_y, n_x = out.shape
            nb_nodes = n_x*n_y
            dx = dy = dt = 1
 
            # run n_simu non-conditional simulation
            I = sparse_eye(nb_nodes)
            x_simu = []
            for ibatch in range(n_b):
                x_simu_ = []
                for it in range(n_t):
                    A = DiffOperator(n_x,
                                     n_y,
                                     dx,
                                     dy,
                                     theta[1][ibatch,:,:,it],
                                     theta[2][ibatch,:,:,:,it],
                                     theta[0][ibatch,:,:,it])
                    M = I+pow_diff_operator(A,pow=2,sparse=True)
                    x_simu_ = self.run_simulation(it,theta[3][ibatch,0,:,it],M,
                                                  x_simu_,dx,dy,dt,n_init_run=10)
                x_simu_ = torch.stack(x_simu_,dim=0)
                x_simu_ = torch.reshape(x_simu_,(n_t,n_x,n_y))
                # x,y -> y,x
                #x_simu_ = torch.permute(x_simu_,(0,2,1))
                x_simu.append(x_simu_)
            x_simu = torch.stack(x_simu,dim=0).to(device)

            # interpolate the simulation based on LSTM-solver
            inputs_simu = x_simu.clone()
            inputs_obs_simu = x_simu.clone()
            mask = batch.input.isfinite()
            inputs_obs_simu[~mask] = 0.
            # create simu_batch 
            simu_batch = batch
            simu_batch = simu_batch._replace(input=inputs_obs_simu)
            simu_batch = simu_batch._replace(tgt=inputs_simu)
            x_itrp_simu, _ = self(batch=simu_batch)
            x_itrp_simu = x_itrp_simu.detach()
                
            # conditional simulation
            x_simu_cond = out+(x_simu-x_itrp_simu)

            self.test_data[i].append(torch.stack(
                [
                batch.input.cpu() * s + m,
                batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
                inputs_simu.cpu() * s + m,
                x_itrp_simu.cpu() * s + m,
                x_simu_cond.squeeze(dim=-1).detach().cpu() * s + m,
                ],
                dim=1,
            ))
        
        # reshape parameters as maps
        kappa = torch.reshape(theta[0],(len(out),1,n_x,n_y,n_t))
        m = torch.reshape(theta[1],(len(out),2,n_x,n_y,n_t))
        H = torch.reshape(theta[2],(len(out),2,2,n_x,n_y,n_t))
        tau = torch.reshape(theta[3],(len(out),1,n_x,n_y,n_t))
        #kappa = torch.permute(kappa,(0,4,3,2,1))
        kappa = torch.permute(kappa,(0,4,2,3,1))
        kappa = kappa[:,:,:,:,0]
        #m = torch.permute(m,(0,4,3,2,1))
        m = torch.permute(m,(0,4,2,3,1))
        m1 = m[:,:,:,:,0]
        m2 = m[:,:,:,:,1]
        #H = torch.permute(H,(0,5,4,3,2,1))
        H = torch.permute(H,(0,5,3,4,1,2))
        H11 = H[:,:,:,:,0,0]
        H12 = H[:,:,:,:,0,1]
        H21 = H[:,:,:,:,1,0] 
        H22 = H[:,:,:,:,1,1]
        #tau = torch.permute(tau,(0,4,3,2,1))
        tau = torch.permute(tau,(0,4,2,3,1))
        tau = tau[:,:,:,:,0]

        self.test_params.append(torch.stack(
            [
                kappa.cpu(),
                m1.cpu(),
                m2.cpu(),
                H11.cpu(),
                H12.cpu(),
                H21.cpu(),
                H22.cpu(),
                tau.cpu()
            ],
            dim=1,
        ))

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'mxy', 'sample_x', 'mx', 'sample_xy']

    @property
    def test_params_quantities(self):
        return ['kappa', 'm1', 'm2', 'H11', 'H12', 'H21', 'H22', 'tau']

    def on_test_epoch_end(self):

        rec_da_wsimu = []
        for i in range(self.n_simu):
            # reconstruct data
            if isinstance(self.trainer.test_dataloaders,list):
                rec_da = self.trainer.test_dataloaders[0].dataset.reconstruct(
                    self.test_data[i], self.rec_weight.cpu().numpy()
                )
            else:
                rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                    self.test_data[i], self.rec_weight.cpu().numpy()
                )
            rec_da = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')
            rec_da_wsimu.append(rec_da)
        
        if len(rec_da_wsimu)>1:
            self.test_data = xr.concat(rec_da_wsimu, pd.Index(np.arange(self.n_simu), name='simu'))
        else:
            self.test_data = rec_da_wsimu[0]

        # reconstruct parameters
        if isinstance(self.trainer.test_dataloaders,list):
            rec_params = self.trainer.test_dataloaders[0].dataset.reconstruct(
                self.test_params, self.rec_weight.cpu().numpy()
            )
        else:
            rec_params = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_params, self.rec_weight.cpu().numpy()
            )

        self.test_params = rec_params.assign_coords(
            dict(v0=self.test_params_quantities)
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
            self.test_params.to_netcdf(Path(self.logger.log_dir) / 'test_params.nc')
            print(Path(self.trainer.log_dir) / 'test_params.nc')
            self.logger.log_metrics(metrics.to_dict())


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
from sksparse.cholmod import cholesky

class Upsampler(nn.Module):
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

class Lit4dVarNet(pl.LightningModule):
    def __init__(self, solver, solver2, rec_weight, optim_weight1, optim_weight2, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, n_simu=100, train_init=True, downsamp = None, frcst_lead = None, epoch_start_opt2=300):

        super().__init__()
        self.solver = solver
        self.solver2 = solver2

        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.register_buffer('optim_weight1', torch.from_numpy(optim_weight1), persistent=persist_rw)
        self.register_buffer('optim_weight2', torch.from_numpy(optim_weight2), persistent=persist_rw)

        # crop_daw for the joint solver of x/theta
        # mapping: [--,--,x,x,x,x,x,--,--]
        # forecast: [--,--,--,--,x,x,x,x,x]
        self.crop_daw = solver.optim_weight1.patch_dims['time']-solver.optim_weight2.patch_dims['time']
        if self.crop_daw != 0:
            if self.frcst_lead is not None:
                self.sel_cropdaw = np.arange(solver.optim_weight1.patch_dims['time']-self.crop_daw-1,solver.optim_weight1.patch_dims['time'])
            else:
                self.sel_cropdaw = np.arange(int(self.crop_daw/2),solver.optim_weight1.patch_dims['time']-int(self.crop_daw/2))
        else:
            self.sel_cropdaw = np.arange(solver.optim_weight1.patch_dims['time'])

        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.n_simu = n_simu

        self.downsamp = downsamp
        self.down = torch.nn.AvgPool2d(downsamp)
        self.up = (
            Upsampler(scale_factor=self.downsamp,
                     mode='bilinear',
                     align_corners=False,
                     antialias=True)
            if downsamp is not None
            else torch.nn.Identity()
        )
        self.epoch_start_opt2 = epoch_start_opt2

        # Cholesky factorization factor
        self.factor = None

        # If forecast:
        self.frcst_lead = frcst_lead

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

    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.current_epoch < self.epoch_start_opt2:
            if optimizer_idx == 1:
                return None
        else:
            if optimizer_idx == 0:
                return None
            if optimizer_idx == 1:
                print("lr", self.lr_schedulers()[1].get_last_lr())

        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def forward(self, batch):
        out = self.solver2(batch=batch)
        if self.current_epoch >= self.epoch_start_opt2:
            # provide mu as coarse version of 4DVarNet outputs
            mean = self.up(self.down(out))
            _, theta = self.solver(batch=batch,
                            x_init=out.detach(),
                            mu=mean.detach())
        else:
            theta = None
        return out, theta

    def step(self, batch, phase=""):
        if batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        loss, out, theta = self.base_step(batch, phase)
        grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.optim_weight)
    
        # prepare initialization of the second solver with classic 4DVarNet
        if self.current_epoch<50:
            prior_cost = self.solver2.prior_cost(self.solver2.init_state(batch, out))
            training_loss = 50*loss  + 1000 * grad_loss + 1.0 * prior_cost
            print(50*loss, 1000 * grad_loss)
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_prior_loss", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
        # training of the augmented state solver
        else:
            if self.solver.aug_state==True:
                nll_loss = torch.nanmean(self.solver.nll(batch.tgt,
                                                 theta = theta,
                                                 mu = self.up(self.down(out)),
                                                 det=True))
            else:
                nll_loss = torch.nanmean(self.solver.nll(batch.tgt,
                                                 theta = solver.nll.encoder(out),
                                                 det=True))
            if torch.isnan(nll_loss)==True:
                return None, None
            #training_loss = 50*loss  + 1000 * grad_loss + nll_loss * 1e-4
            training_loss = 50*loss + nll_loss * 1e-6
            print(50*loss, nll_loss * 1e-6)
            self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log( f"{phase}_nll_loss", nll_loss*1e-6, prog_bar=True, on_step=False, on_epoch=True)
   
        return training_loss, out

    def base_step(self,batch,phase=""):

        out, theta = self(batch=batch)

        # mse loss
        loss = self.weighted_mse(out - batch.tgt, self.optim_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out, theta

    def configure_optimizers(self):
        return self.opt_fn(self,epoch_start_opt2=self.epoch_start_opt2)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
            self.test_params = []
            self.test_simu =[]

        # get norm_stats
        m, s = self.norm_stats

        batch_ = batch    
        new_input = (batch_.input).nan_to_num()
        if (self.frcst_lead is not None) and (self.frcst_lead>0): 
            new_input[:,(-self.frcst_lead):,:,:] = 0.
        batch_ = batch_._replace(input=new_input.to(device))
        batch_ = batch_._replace(tgt=batch_.tgt.to(device))
        
        # 4DVarNets scheme
        out, theta = self(batch=batch_)
        n_b, n_t, n_y, n_x = out.shape
        nb_nodes = n_x*n_y
        dx = dy = dt = 1
 
        self.test_data.append(torch.stack(
            [
                batch_.input.cpu() * s + m,
                batch_.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

        # run n_simu non-conditional simulations
        if self.solver.nll.downsamp is not None: 
            theta[0],theta[1],theta[2],theta[3] = self.solver.nll.downsamp_params(theta[0],theta[1],theta[2],theta[3],
                                                             sp_dims=[n_y, n_x])
            

        Q = self.solver.nll.operator_spde(theta[0],theta[1],theta[2],theta[3],
                                          store_block_diag=False)
    
        x_simu = []
        for ibatch in range(n_b):
            Q_ = Q[ibatch].detach().cpu()
            Q_sp = sparse_torch2scipy(Q_)
            if self.factor is None:
                self.factor = cholesky(Q_sp,ordering_method='natural')
            else:
                self.factor.cholesky_inplace(Q_sp)
            if self.solver.nll.downsamp is not None:
                RM = self.factor.apply_P(torch.randn((nb_nodes//(self.solver.nll.downsamp**2))*n_t,self.n_simu))
            else:
                RM = self.factor.apply_P(torch.randn(nb_nodes*n_t,self.n_simu))
            x_simu_ = torch.FloatTensor(self.factor.solve_Lt(RM,
                                                        use_LDLt_decomposition=False)).to(device)
            if self.solver.nll.downsamp is not None:
                x_simu_ = torch.squeeze(torch.stack([self.solver.nll.up(torch.reshape(x_simu_[:,i], 
                                                     (1,n_t,
                                                     n_x//self.solver.nll.downsamp,
                                                     n_y//self.solver.nll.downsamp))) for i in range(self.n_simu)],dim=4),
                                         dim=0)
            else:
                x_simu_ = torch.reshape(x_simu_,(n_t,n_x,n_y,n_simu))
            x_simu.append(x_simu_)
        x_simu = torch.stack(x_simu,dim=0).to(device)
    
        if self.solver.nll.downsamp is not None:
            theta[0],theta[1],theta[2],theta[3]  = self.solver.nll.upsamp_params(theta[0],theta[1],theta[2],theta[3] ,
                                                              sp_dims=[n_y//self.solver.nll.downsamp, 
                                                                       n_x//self.solver.nll.downsamp])
    
        # interpolate the simulation based on LSTM-self.solver
        x_simu_cond = []
        x_simu_itrp = []
        for i in range(self.n_simu):
            mean = self.up(self.down(out))
            # non-conditional simulations = mean + anomaly 
            inputs_simu = mean + x_simu[:,:,:,:,i]
            inputs_obs_simu = mean + x_simu[:,:,:,:,i]
            mask = batch.input.isfinite()
            inputs_obs_simu[~mask] = 0.
            # create simu_batch
            simu_batch = batch
            simu_batch = simu_batch._replace(input=inputs_obs_simu.nan_to_num())
            simu_batch = simu_batch._replace(tgt=inputs_simu.nan_to_num())
            # itrp non-conditional simulations
            #x_itrp_simu, _ = self(simu_batch)
            x_itrp_simu = self.solver2(simu_batch)
            #mean = self.up(self.down(x_itrp_simu))
            #x_itrp_simu, _ = self.solver(simu_batch,
            #                             x_init=x_itrp_simu.detach(),
            #                             mu=mean.detach())
            x_simu_itrp.append(x_itrp_simu)
            # conditional simulations
            x_simu_cond_ = (simu_batch.tgt - x_itrp_simu) + out
            x_simu_cond.append(x_simu_cond_)
        x_simu_itrp = torch.stack(x_simu_itrp,dim=4).to(device).detach().cpu()                
        x_simu_cond = torch.stack(x_simu_cond,dim=4).to(device).detach().cpu()
        
        self.test_simu.append(torch.stack(
                [
                    x_simu.detach().cpu(),
                    x_simu_itrp,
                    x_simu_cond * s + m,
                ],
                dim=1,
            ))  
        
        # reshape parameters as maps
        kappa = torch.reshape(theta[0],(len(out),1,n_x,n_y,n_t))
        ma = torch.reshape(theta[1],(len(out),2,n_x,n_y,n_t))
        H = torch.reshape(theta[2],(len(out),2,2,n_x,n_y,n_t))
        tau = torch.reshape(theta[3],(len(out),1,n_x,n_y,n_t))
        kappa = torch.permute(kappa,(0,4,2,3,1))
        kappa = kappa[:,:,:,:,0]
        ma = torch.permute(ma,(0,4,2,3,1))
        m1 = ma[:,:,:,:,0]
        m2 = ma[:,:,:,:,1]
        H = torch.permute(H,(0,5,3,4,1,2))
        H11 = H[:,:,:,:,0,0]
        H12 = H[:,:,:,:,0,1]
        H21 = H[:,:,:,:,1,0]
        H22 = H[:,:,:,:,1,1]
        tau = torch.permute(tau,(0,4,2,3,1))
        tau = tau[:,:,:,:,0]
        
        self.test_params.append(torch.stack(
                [
                    kappa.detach().cpu(),
                    m1.detach().cpu(),
                    m2.detach().cpu(),
                    H11.detach().cpu(),
                    H12.detach().cpu(),
                    H21.detach().cpu(),
                    H22.detach().cpu(),
                    tau.detach().cpu()
                ],
                dim=1,
            ))  

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'out']

    @property
    def test_simu_quantities(self):
        return ['sample_x', 'mx', 'sample_xy']

    @property
    def test_params_quantities(self):
        return ['kappa', 'm1', 'm2', 'H11', 'H12', 'H21', 'H22', 'tau']

    '''
    def on_train_epoch_end(self):
        if self.train_init:
            torch.save(self.solver2.state_dict(),
                       '/homes/m19beauc/4dvarnet-starter/ckpt/ckptnew_spde_wonll_rzf=2_frcst'+str(self.frcst_lead)+'.pth')
    '''

    def on_test_epoch_end(self):

        # reconstruct mean
        if isinstance(self.trainer.test_dataloaders,list):
            rec_da = self.trainer.test_dataloaders[0].dataset.reconstruct(
                    self.test_data, self.rec_weight.cpu().numpy()
            )
        else:
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                    self.test_data, self.rec_weight.cpu().numpy()
            )
        self.test_data = rec_da.assign_coords(
                    dict(v0=self.test_quantities)
                ).to_dataset(dim='v0')

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

        # reconstruct simulations
        rec_da_wsimu = []
        for i in range(self.n_simu):
            if isinstance(self.trainer.test_dataloaders,list):
                rec_simu = self.trainer.test_dataloaders[0].dataset.reconstruct(
                    [ts[:,:,:,:,:,i] for ts in self.test_simu], self.rec_weight.cpu().numpy()
                )
            else:
                rec_simu = self.trainer.test_dataloaders.dataset.reconstruct(
                    [ts[:,:,:,:,:,i] for ts in self.test_simu], self.rec_weight.cpu().numpy()
                )
            rec_da_wsimu.append(rec_simu)
        if len(rec_da_wsimu)>1:
            self.test_simu = xr.concat(rec_da_wsimu, pd.Index(np.arange(self.n_simu), name='simu'))
        else:
            self.test_simu = rec_da_wsimu[0]
    
        self.test_simu = self.test_simu.assign_coords(
                    dict(v0=self.test_simu_quantities)
                ).to_dataset(dim='v0')

        self.test_data = xr.merge([self.test_data,self.test_simu,self.test_params])
        self.test_data = self.test_data.update({'std_simu':(('time','lat','lon'),
                                        self.test_data.sample_xy.std(dim='simu').values)})

        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')
            print(Path(self.trainer.log_dir) / 'test_data.nc')

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.logger.log_metrics(metrics.to_dict())



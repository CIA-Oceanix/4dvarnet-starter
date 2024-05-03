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
import einops

def smooth(field):
    field = field.data
    m1 = torch.nn.AvgPool2d((2,2))
    m2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    field = m2(m1(field))
    return field

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
    def __init__(self, solver, solver2, rec_weight, optim_weight1, optim_weight2, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, n_simu=100, downsamp = None, frcst_lead = None, epoch_start_opt2=1000,start_simu_idx=0,ncfile_name='test_data.nc'):

        super().__init__()
        self.solver = solver
        self.solver2 = solver2

        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.register_buffer('optim_weight1', torch.from_numpy(optim_weight1), persistent=persist_rw)
        self.register_buffer('optim_weight2', torch.from_numpy(optim_weight2), persistent=persist_rw)

        # If forecast:
        self.frcst_lead = frcst_lead

        # crop_daw for the joint solver of x/theta
        # mapping: [--,--,x,x,x,x,x,--,--]
        # forecast: [--,--,--,--,x,x,x,x,x]
        self.crop_daw = optim_weight1.shape[0]-optim_weight2.shape[0]
        if self.crop_daw != 0:
            if self.frcst_lead is not None:
                self.sel_crop_daw = np.arange(self.crop_daw,optim_weight1.shape[0])
            else:
                self.sel_crop_daw = np.arange(int(self.crop_daw/2),optim_weight1.shape[0]-int(self.crop_daw/2))
        else:
            self.sel_crop_daw = np.arange(optim_weight1.shape[0])

        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)
        self.n_simu = n_simu
        self.start_simu_idx = start_simu_idx

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
        self.ncfile_name = ncfile_name
        # Cholesky factorization factor
        self.factor = None

        # parameter used for outputs
        self.use_gt = True
        self.out_as_first_guess = True

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

    def modify_batch(self,batch):
        batch_ = batch
        new_input = (batch_.input).nan_to_num()
        if (self.frcst_lead is not None) and (self.frcst_lead>0):
            new_input[:,(-self.frcst_lead):,:,:] = 0.
        batch_ = batch_._replace(input=new_input.to(device))
        batch_ = batch_._replace(tgt=batch_.tgt.nan_to_num().to(device))
        return batch_

    def crop_batch(self, batch):
        cropped_batch = batch
        cropped_batch = cropped_batch._replace(input=(cropped_batch.input[:,self.sel_crop_daw,:,:]).nan_to_num().to(device))
        cropped_batch = cropped_batch._replace(tgt=(cropped_batch.tgt[:,self.sel_crop_daw,:,:]).nan_to_num().to(device))
        return cropped_batch

    def corrupt_batch(self, batch, out, niter=3):
        corrupted_batch = batch
        for _ in range(niter):
            corrupted_batch = corrupted_batch._replace(tgt=out.clone())
            mask = (batch.input!=0)
            out[~mask] = 0.   
            corrupted_batch = corrupted_batch._replace(input=out.clone())
            out = self.solver2(corrupted_batch)
        return out

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

    def forward(self, batch, phase=""):

        batch_ = self.modify_batch(batch)
        out = self.solver2(batch=batch_)
        # provide mu as coarse version of 4DVarNet outputs
        corrupted_out = self.corrupt_batch(batch_, out.clone())

        if ((self.current_epoch >= self.epoch_start_opt2) or (phase=="test")):
            cropped_batch = self.crop_batch(batch_)
            if not self.use_gt:
                out_spde, theta = self.solver(batch=cropped_batch, 
                                x_init=out[:,self.sel_crop_daw,:,:].detach(),
                                mu=corrupted_out[:,self.sel_crop_daw,:,:].detach())
            else:
                out_spde, theta = self.solver(batch=cropped_batch,
                                x_init=batch_.tgt[:,self.sel_crop_daw,:,:].detach(),
                                mu=out[:,self.sel_crop_daw,:,:].detach())
                corrupted_out = out
            if not self.out_as_first_guess:
                out = out_spde
            else:
                out = out[:,self.sel_crop_daw,:,:]
        else:
            theta = None
        return out, corrupted_out, theta

    def step(self, batch, phase=""):
        if batch.tgt.isfinite().float().mean() < 0.9:
            return None, None

        batch = self.modify_batch(batch)
        loss, out, corrupted_out, theta = self.base_step(batch, phase)
    
        # prepare initialization of the second solver with classic 4DVarNet
        if self.current_epoch<self.epoch_start_opt2:
            grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.optim_weight1)
            prior_cost = self.solver2.prior_cost(self.solver2.init_state(batch, out))
            training_loss = 50*loss  + 1000 * grad_loss + 1.0 * prior_cost
            print(50*loss, 1000 * grad_loss)
            with torch.no_grad():
                self.log( f"{phase}_gloss", grad_loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log( f"{phase}_prior_loss", prior_cost, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)

        # training of the augmented state solver
        else:
            if self.solver.aug_state==True:
                nll_loss = torch.nanmean(self.solver.nll(self.crop_batch(batch).tgt,
                                                 theta = theta,
                                                 mu = corrupted_out[:,self.sel_crop_daw,:,:].detach(),
                                                 det=True))
            else:
                nll_loss = torch.nanmean(self.solver.nll(self.crop_batch(batch).tgt,
                                                 theta = solver.nll.encoder(out),
                                                 det=True))
            if torch.isnan(nll_loss)==True:
                return None, None
            training_loss = 10*loss + nll_loss * 1e-6
            print(10*loss, nll_loss * 1e-6)
            with torch.no_grad():
                self.log( f"{phase}_nll_loss", nll_loss*1e-6, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f"{phase}_mse", (10000 * loss * self.norm_stats[1]**2) + 1e-3*nll_loss, prog_bar=True, on_step=False, on_epoch=True)
   
        return training_loss, out

    def base_step(self,batch,phase=""):

        out, corrupted_out, theta = self(batch=batch, phase="")
        # mse loss
        if self.current_epoch<self.epoch_start_opt2:
            loss = self.weighted_mse(out - batch.tgt, self.optim_weight1)
        else:
            loss = self.weighted_mse(out - self.crop_batch(batch).tgt, self.optim_weight2)

        with torch.no_grad():
            #self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out, corrupted_out, theta

    def configure_optimizers(self):
        return self.opt_fn(self,epoch_start_opt2=self.epoch_start_opt2)

    def test_step(self, batch, batch_idx, phase="test"):
        if batch_idx == 0:
            self.test_data = []
            self.test_params = []
            self.test_simu =[]

        # get norm_stats
        m, s = self.norm_stats

        batch_ = self.modify_batch(batch)

        # 4DVarNets scheme
        out, corrupted_out, theta = self(batch=batch, phase=phase)
        n_b, n_t, n_y, n_x = out.shape
        nb_nodes = n_x*n_y
        dx = dy = dt = 1
 
        self.test_data.append(torch.stack(
            [
                self.crop_batch(batch_).input.cpu() * s + m,
                self.crop_batch(batch_).tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

        # run n_simu non-conditional simulations
        if self.solver.nll.downsamp is not None: 
            theta[0],theta[1],theta[2],theta[3] = self.solver.nll.downsamp_params(theta[0],theta[1],theta[2],theta[3],
                                                             sp_dims=[n_y, n_x])
            

        if batch_idx/self.trainer.num_test_batches[0] >=self.start_simu_idx:

            print(batch_idx)
            print(self.trainer.num_test_batches[0])
            print(self.start_simu_idx)

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
                    x_simu_ = torch.reshape(x_simu_,(n_t,n_x,n_y,self.n_simu))
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
                mean = corrupted_out[:,self.sel_crop_daw,:,:]
                inputs_simu = mean + smooth(x_simu[:,:,:,:,i])
                inputs_obs_simu = mean + smooth(x_simu[:,:,:,:,i])
                # increase size of simu batch
                if self.frcst_lead is not None:
                    inputs_simu = torch.cat((einops.repeat(inputs_simu[:,0,:,:], 'b y x -> b t y x', t=self.crop_daw),inputs_simu),dim=1)
                    inputs_obs_simu = torch.cat((einops.repeat(inputs_simu[:,0,:,:], 'b y x -> b t y x', t=self.crop_daw),inputs_obs_simu),dim=1)
                else:
                    inputs_simu = torch.cat((einops.repeat(inputs_simu[:,0,:,:], 'b y x -> b t y x', t=int(self.crop_daw//2)),
                             inputs_simu,
                             einops.repeat(inputs_simu[:,-1,:,:], 'b y x -> b t y x', t=int(self.crop_daw//2))),dim=1)
                    inputs_obs_simu = torch.cat((einops.repeat(inputs_simu[:,0,:,:], 'b y x -> b t y x', t=int(self.crop_daw//2)),
                             inputs_obs_simu,
                             einops.repeat(inputs_simu[:,-1,:,:], 'b y x -> b t y x', t=int(self.crop_daw//2))),dim=1)
                mask = (batch_.input!=0)
                inputs_obs_simu[~mask] = 0.
                simu_batch = batch_
                simu_batch = simu_batch._replace(input=inputs_obs_simu)
                simu_batch = simu_batch._replace(tgt=inputs_simu)
                # itrp non-conditional simulations
                #x_itrp_simu, _ = self(simu_batch)
                x_itrp_simu = self.solver2(simu_batch)
                x_itrp_simu = x_itrp_simu[:,self.sel_crop_daw,:,:]
                x_simu_itrp.append(x_itrp_simu)
                # conditional simulations
                x_simu_cond_ = (simu_batch.tgt[:,self.sel_crop_daw,:,:] - x_itrp_simu) + out.detach()
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
        else:
            self.test_simu.append(torch.stack(
            [
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
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

        out = None
        x_simu = None
        x_simu_itrp = None
        x_simu_cond = None
        theta = None
        kappa = None
        tau = None
        m = None
        H = None

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'out']

    @property
    def test_simu_quantities(self):
        return ['sample_x', 'mx', 'sample_xy']

    @property
    def test_params_quantities(self):
        return ['kappa', 'm1', 'm2', 'H11', 'H12', 'H21', 'H22', 'tau']

    def on_test_epoch_end(self):

        # reconstruct mean
        if isinstance(self.trainer.test_dataloaders,list):
            rec_da = self.trainer.test_dataloaders[0].dataset.reconstruct(
                    self.test_data, self.rec_weight.cpu().numpy(), crop = self.crop_daw
            )
        else:
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                    self.test_data, self.rec_weight.cpu().numpy(), crop = self.crop_daw
            )
        self.test_data = rec_da.assign_coords(
                    dict(v0=self.test_quantities)
                ).to_dataset(dim='v0')

        # reconstruct parameters
        if isinstance(self.trainer.test_dataloaders,list):
            rec_params = self.trainer.test_dataloaders[0].dataset.reconstruct(
                    self.test_params, self.rec_weight.cpu().numpy(), crop = self.crop_daw
            )
        else:
            rec_params = self.trainer.test_dataloaders.dataset.reconstruct(
                        self.test_params, self.rec_weight.cpu().numpy(), crop = self.crop_daw
            )
        self.test_params = rec_params.assign_coords(
                    dict(v0=self.test_params_quantities)
                ).to_dataset(dim='v0')

        # reconstruct simulations
        rec_da_wsimu = []
        for i in range(self.n_simu):
            if isinstance(self.trainer.test_dataloaders,list):
                rec_simu = self.trainer.test_dataloaders[0].dataset.reconstruct(
                    [ts[:,:,:,:,:,i] for ts in self.test_simu], self.rec_weight.cpu().numpy(), crop = self.crop_daw
                )
            else:
                rec_simu = self.trainer.test_dataloaders.dataset.reconstruct(
                    [ts[:,:,:,:,:,i] for ts in self.test_simu], self.rec_weight.cpu().numpy(), crop = self.crop_daw
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
            self.test_data.to_netcdf(Path(self.logger.log_dir) / self.ncfile_name)
            print(Path(self.trainer.log_dir) / self.ncfile_name)

        metric_data = self.test_data.pipe(self.pre_metric_fn)
        metrics = pd.Series({
            metric_n: metric_fn(metric_data) 
            for metric_n, metric_fn in self.metrics.items()
        })

        print(metrics.to_frame(name="Metrics").to_markdown())
        if self.logger:
            self.logger.log_metrics(metrics.to_dict())


class Lit4dVarNet_wcov(Lit4dVarNet):

    def modify_batch(self,batch):
        batch_ = batch
        new_input = (batch_.input).nan_to_num()
        if (self.frcst_lead is not None) and (self.frcst_lead>0):
            new_input[:,(-self.frcst_lead):,:,:] = 0.
        batch_ = batch_._replace(input=new_input.to(device))
        batch_ = batch_._replace(tgt=batch_.tgt.nan_to_num().to(device))
        batch_ = batch_._replace(u=batch_.u.nan_to_num().to(device))
        batch_ = batch_._replace(v=batch_.v.nan_to_num().to(device))
        return batch_


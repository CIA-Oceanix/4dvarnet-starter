import pandas as pd
#import matplotlib.pyplot as plt
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
    def __init__(self, solver, solver2, rec_weight, optim_weight1, optim_weight2, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True, n_simu=100, downsamp = None, frcst_lead = None, use_gt = True, out_as_first_guess=True, type_loss="ssh_sst", epoch_start_opt2=1000,start_simu_idx=0,ncfile_name='test_data.nc'):

        super().__init__()
        self.solver = solver
        self.solver2 = solver2

        self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        self.register_buffer('optim_weight1', torch.from_numpy(optim_weight1), persistent=persist_rw)
        self.register_buffer('optim_weight2', torch.from_numpy(optim_weight2), persistent=persist_rw)

        # If forecast:
        self.frcst_lead = frcst_lead

        # Loss weighting
        self.type_loss = type_loss

        # crop_daw for the joint solver of x/theta
        # mapping: [--,--,x,x,x,x,x,--,--]
        # forecast: [--,--,--,--,x,x,x,x,x]
        self.crop_daw = optim_weight1.shape[0]-optim_weight2.shape[0]
        if self.crop_daw != 0:
            if self.frcst_lead is not None:
                self.sel_crop_daw = np.concatenate((np.arange(self.crop_daw//2,optim_weight1.shape[0]//2),
                                                    np.arange(optim_weight1.shape[0]//2 + self.crop_daw//2,optim_weight1.shape[0])))
            else:
                self.sel_crop_daw = np.concatenate((np.arange(int(self.crop_daw/4),optim_weight1.shape[0]//2-int(self.crop_daw/4)),
                                                    np.arange(optim_weight1.shape[0]//2 + int(self.crop_daw/4),optim_weight1.shape[0]-int(self.crop_daw/4))))
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
        self.use_gt = use_gt
        self.out_as_first_guess = out_as_first_guess

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1., 0., 1.)

    @staticmethod
    def weighted_mse(err, weight, type_loss):
        n_b, n_t, n_y, n_x = err.shape
        n_t = n_t//2

        err_w1 = err[:, :n_t, :, :] * weight[None, :n_t, :, :]
        err_w2 = err[:, n_t:, :, :] * weight[None, n_t:, :, :]
        if type_loss=="ssh_sst":
            err_w = torch.cat((err_w1,err_w2),dim=1)
        elif type_loss=="ssh":
            err_w = torch.cat((10*err_w1,err_w2),dim=1)
        else:
            err_w = torch.cat((err_w1,10*err_w2),dim=1)

        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def modify_batch(self,batch):
        batch_ = batch
        n_t = batch.tgt.shape[1]
        n_t = n_t//2
        new_input = (batch_.input).nan_to_num()
        if (self.frcst_lead is not None) and (self.frcst_lead>0):
            new_input[:,(n_t-self.frcst_lead):n_t,:,:] = 0.
            new_input[:,((2*n_t)-self.frcst_lead):,:,:] = 0.
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
                cropped_batch = cropped_batch._replace(input=cropped_batch.tgt.to(device)) 
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
            grad_loss = self.weighted_mse( kfilts.sobel(out) - kfilts.sobel(batch.tgt), self.optim_weight1, self.type_loss)
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
                theta_ssh = theta[:4]
                theta_sst = theta[4:]
                if self.type_loss=='ssh_sst':
                    w_nll = [1e-6,1e-6]
                elif self.type_loss=='ssh':
                    w_nll = [1e-6,1e-7]
                else:
                    w_nll = [1e-7,1e-6]
                nll_loss_ssh = w_nll[0]*torch.nanmean(self.solver.nll(self.crop_batch(batch).tgt[:,:(self.optim_weight2.shape[0]//2),:,:],
                                                 theta = theta_ssh,
                                                 mu = (corrupted_out[:,self.sel_crop_daw,:,:])[:,:(self.optim_weight2.shape[0]//2),:,:].detach(),
                                                 det=True)) 
                nll_loss_sst = w_nll[1]*torch.nanmean(self.solver.nll(self.crop_batch(batch).tgt[:,(self.optim_weight2.shape[0]//2):,:,:],
                                                 theta = theta_sst,
                                                 mu = (corrupted_out[:,self.sel_crop_daw,:,:])[:,(self.optim_weight2.shape[0]//2):,:,:].detach(),
                                                 det=True))
                nll_loss = nll_loss_ssh + nll_loss_sst
            else:
                nll_loss = torch.nanmean(self.solver.nll(self.crop_batch(batch).tgt[:,:(self.optim_weight2.shape[0]//2),:,:],
                                                 theta = solver.nll.encoder(out),
                                                 det=True))
            if torch.isnan(nll_loss)==True:
                return None, None
            training_loss = 10*loss + nll_loss 
            print(10*loss, nll_loss, nll_loss_ssh, nll_loss_sst)
            with torch.no_grad():
                self.log( f"{phase}_nll_loss", nll_loss*1e-6, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f"{phase}_mse", (10000 * loss * self.norm_stats[1]**2) + 1e-3*nll_loss, prog_bar=True, on_step=False, on_epoch=True)
   
        return training_loss, out

    def base_step(self,batch,phase=""):

        out, corrupted_out, theta = self(batch=batch, phase="")
        # mse loss
        if self.current_epoch<self.epoch_start_opt2:
            loss = self.weighted_mse(out - batch.tgt, self.optim_weight1, self.type_loss)
        else:
            loss = self.weighted_mse(out - self.crop_batch(batch).tgt, self.optim_weight2, self.type_loss)

        with torch.no_grad():
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
        m_ssh, s_ssh, m_sst, s_sst = self.norm_stats

        batch_ = self.modify_batch(batch)

        '''
        # plot
        data = xr.Dataset(data_vars=dict(
                               inp=(["time", "lat", "lon"], batch_.input[0].detach().cpu().data),
                               tgt=(["time", "lat", "lon"], batch_.tgt[0].detach().cpu().data)
                               ),
                          coords=(dict(time=range(18),
                                  lon=range(120),
                                  lat=range(120))))
        data.inp.plot(col='time',col_wrap=9)
        plt.show()
        '''

        # 4DVarNets scheme
        out, corrupted_out, theta = self(batch=batch, phase=phase)
        theta_ssh = theta[:4]
        theta_sst = theta[4:]
        n_b, n_t, n_y, n_x = out.shape
        n_t = n_t//2
        nb_nodes = n_x*n_y
        dx = dy = dt = 1
 
        self.test_data.append(torch.stack(
            [
                self.crop_batch(batch_).input.cpu()[:,:(self.optim_weight2.shape[0]//2),:,:] * s_ssh + m_ssh,
                self.crop_batch(batch_).tgt.cpu()[:,:(self.optim_weight2.shape[0]//2),:,:] * s_ssh + m_ssh,
                out.squeeze(dim=-1).detach().cpu()[:,:(self.optim_weight2.shape[0]//2),:,:] * s_ssh + m_ssh,
                self.crop_batch(batch_).input.cpu()[:,(self.optim_weight2.shape[0]//2):,:,:] * s_sst + m_sst,
                self.crop_batch(batch_).tgt.cpu()[:,(self.optim_weight2.shape[0]//2):,:,:] * s_sst + m_sst,
                out.squeeze(dim=-1).detach().cpu()[:,(self.optim_weight2.shape[0]//2):,:,:] * s_sst + m_sst,
            ],
            dim=1,
        ))

        # run n_simu non-conditional simulations
        if self.solver.nll.downsamp is not None: 
            theta[0],theta[1],theta[2],theta[3] = self.solver.nll.downsamp_params(theta[0],theta[1],theta[2],theta[3],
                                                             sp_dims=[n_y, n_x])
            theta[4],theta[5],theta[6],theta[7] = self.solver.nll.downsamp_params(theta[4],theta[5],theta[6],theta[7],
                                                             sp_dims=[n_y, n_x])
           

        if batch_idx/self.trainer.num_test_batches[0] >=self.start_simu_idx:

            def run_simu(Q):
    
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

                return x_simu

            # simu ssh
            Q_ssh = self.solver.nll.operator_spde(theta[0],theta[1],theta[2],theta[3],
                                          store_block_diag=False)
            x_simu_ssh = run_simu(Q_ssh)
            if self.solver.nll.downsamp is not None:
                theta[0],theta[1],theta[2],theta[3]  = self.solver.nll.upsamp_params(theta[0],theta[1],theta[2],theta[3] ,
                                                              sp_dims=[n_y//self.solver.nll.downsamp,
                                                                       n_x//self.solver.nll.downsamp])
            # simu sst
            Q_sst = self.solver.nll.operator_spde(theta[4],theta[5],theta[6],theta[7],
                                          store_block_diag=False)
            x_simu_sst = run_simu(Q_sst)
            if self.solver.nll.downsamp is not None:
                theta[4],theta[5],theta[6],theta[7]  = self.solver.nll.upsamp_params(theta[4],theta[5],theta[6],theta[7] ,
                                                              sp_dims=[n_y//self.solver.nll.downsamp,
                                                                       n_x//self.solver.nll.downsamp])

            x_simu = torch.cat((x_simu_ssh,x_simu_sst),dim=1)
    
            # interpolate the simulation based on LSTM-self.solver
            x_simu_cond = []
            x_simu_itrp = []
            for i in range(self.n_simu):
                mean = corrupted_out[:,self.sel_crop_daw,:,:]
                inputs_simu = mean + smooth(x_simu[:,:,:,:,i])
                inputs_simu_ssh = inputs_simu[:,:(self.optim_weight2.shape[0]//2),:,:]
                inputs_simu_sst = inputs_simu[:,(self.optim_weight2.shape[0]//2):,:,:]
                inputs_obs_simu = mean + smooth(x_simu[:,:,:,:,i])
                inputs_obs_simu_ssh = inputs_obs_simu[:,:(self.optim_weight2.shape[0]//2),:,:]
                inputs_obs_simu_sst = inputs_obs_simu[:,(self.optim_weight2.shape[0]//2):,:,:]
                # increase size of simu batch
                if self.frcst_lead is not None:
                    inputs_simu = torch.cat((einops.repeat(inputs_simu_ssh[:,0,:,:], 'b y x -> b t y x', t=self.crop_daw//2),
                                             inputs_simu_ssh,
                                             einops.repeat(inputs_simu_sst[:,0,:,:], 'b y x -> b t y x', t=self.crop_daw//2),
                                             inputs_simu_sst),dim=1)
                    inputs_obs_simu = torch.cat((einops.repeat(inputs_obs_simu_ssh[:,0,:,:], 'b y x -> b t y x', t=self.crop_daw//2),
                                             inputs_obs_simu_ssh,
                                             einops.repeat(inputs_obs_simu_sst[:,0,:,:], 'b y x -> b t y x', t=self.crop_daw//2),
                                             inputs_obs_simu_sst),dim=1)
                else:
                    inputs_simu = torch.cat((einops.repeat(inputs_simu_ssh[:,0,:,:], 'b y x -> b t y x', t=int(self.crop_daw//4)),
                                             inputs_simu_ssh,
                                             einops.repeat(inputs_simu_ssh[:,-1,:,:], 'b y x -> b t y x', t=int(self.crop_daw//4)),
                                             einops.repeat(inputs_simu_sst[:,0,:,:], 'b y x -> b t y x', t=int(self.crop_daw//4)),
                                             inputs_simu_sst,
                                             einops.repeat(inputs_simu_sst[:,-1,:,:], 'b y x -> b t y x', t=int(self.crop_daw//4))),dim=1)
                    inputs_obs_simu = torch.cat((einops.repeat(inputs_obs_simu_ssh[:,0,:,:], 'b y x -> b t y x', t=int(self.crop_daw//4)),
                                             inputs_obs_simu_ssh,
                                             einops.repeat(inputs_obs_simu_ssh[:,-1,:,:], 'b y x -> b t y x', t=int(self.crop_daw//4)),
                                             einops.repeat(inputs_obs_simu_sst[:,0,:,:], 'b y x -> b t y x', t=int(self.crop_daw//4)),
                                             inputs_obs_simu_sst,
                                             einops.repeat(inputs_obs_simu_sst[:,-1,:,:], 'b y x -> b t y x', t=int(self.crop_daw//4))),dim=1)
                mask = (batch_.input!=0)
                inputs_obs_simu[~mask] = 0.
                simu_batch = batch_
                simu_batch = simu_batch._replace(input=inputs_obs_simu)
                simu_batch = simu_batch._replace(tgt=inputs_simu)
                # itrp non-conditional simulations
                x_itrp_simu = self.solver2(simu_batch)
                x_itrp_simu = x_itrp_simu[:,self.sel_crop_daw,:,:]
                x_simu_itrp.append(x_itrp_simu)
                # conditional simulations
                x_simu_cond_ = (simu_batch.tgt[:,self.sel_crop_daw,:,:] - x_itrp_simu) + out.detach()
                x_simu_cond.append(x_simu_cond_)
            x_simu_itrp = torch.stack(x_simu_itrp,dim=4).to(device).detach().cpu()                
            x_simu_cond = torch.stack(x_simu_cond,dim=4).to(device).detach().cpu()

            x_simu_itrp_ssh = x_simu_itrp[:,:(self.optim_weight2.shape[0]//2),:,:]
            x_simu_itrp_sst = x_simu_itrp[:,(self.optim_weight2.shape[0]//2):,:,:]
            x_simu_cond_ssh = x_simu_cond[:,:(self.optim_weight2.shape[0]//2),:,:]
            x_simu_cond_sst = x_simu_cond[:,(self.optim_weight2.shape[0]//2):,:,:]
    
            self.test_simu.append(torch.stack(
                    [
                        x_simu_ssh.detach().cpu(),
                        x_simu_itrp_ssh,
                        x_simu_cond_ssh * s_ssh + m_ssh,
                        x_simu_sst.detach().cpu(),
                        x_simu_itrp_sst,
                        x_simu_cond_sst * s_sst + m_sst
                    ],
                    dim=1,
                ))
        else:
            self.test_simu.append(torch.stack(
            [
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
                torch.zeros(n_b, n_t, n_y, n_x, self.n_simu),
            ],
            dim=1,
            ))
        
        # reshape parameters as maps
        kappa_ssh = torch.reshape(theta_ssh[0],(len(out),1,n_x,n_y,n_t))
        ma_ssh = torch.reshape(theta_ssh[1],(len(out),2,n_x,n_y,n_t))
        H_ssh = torch.reshape(theta_ssh[2],(len(out),2,2,n_x,n_y,n_t))
        tau_ssh = torch.reshape(theta_ssh[3],(len(out),1,n_x,n_y,n_t))
        kappa_ssh = torch.permute(kappa_ssh,(0,4,2,3,1))
        kappa_ssh = kappa_ssh[:,:,:,:,0]
        ma_ssh = torch.permute(ma_ssh,(0,4,2,3,1))
        m1_ssh = ma_ssh[:,:,:,:,0]
        m2_ssh = ma_ssh[:,:,:,:,1]
        H_ssh = torch.permute(H_ssh,(0,5,3,4,1,2))
        H11_ssh = H_ssh[:,:,:,:,0,0]
        H12_ssh = H_ssh[:,:,:,:,0,1]
        H21_ssh = H_ssh[:,:,:,:,1,0]
        H22_ssh = H_ssh[:,:,:,:,1,1]
        tau_ssh = torch.permute(tau_ssh,(0,4,2,3,1))
        tau_ssh = tau_ssh[:,:,:,:,0]
        # reshape parameters as maps
        kappa_sst = torch.reshape(theta_sst[0],(len(out),1,n_x,n_y,n_t))
        ma_sst = torch.reshape(theta_sst[1],(len(out),2,n_x,n_y,n_t))
        H_sst = torch.reshape(theta_sst[2],(len(out),2,2,n_x,n_y,n_t))
        tau_sst = torch.reshape(theta_sst[3],(len(out),1,n_x,n_y,n_t))
        kappa_sst = torch.permute(kappa_sst,(0,4,2,3,1))
        kappa_sst = kappa_sst[:,:,:,:,0]
        ma_sst = torch.permute(ma_sst,(0,4,2,3,1))
        m1_sst = ma_sst[:,:,:,:,0]
        m2_sst = ma_sst[:,:,:,:,1]
        H_sst = torch.permute(H_sst,(0,5,3,4,1,2))
        H11_sst = H_sst[:,:,:,:,0,0]
        H12_sst = H_sst[:,:,:,:,0,1]
        H21_sst = H_sst[:,:,:,:,1,0]
        H22_sst = H_sst[:,:,:,:,1,1]
        tau_sst = torch.permute(tau_sst,(0,4,2,3,1))
        tau_sst = tau_sst[:,:,:,:,0]

        self.test_params.append(torch.stack(
                [
                    kappa_ssh.detach().cpu(),
                    m1_ssh.detach().cpu(),
                    m2_ssh.detach().cpu(),
                    H11_ssh.detach().cpu(),
                    H12_ssh.detach().cpu(),
                    H21_ssh.detach().cpu(),
                    H22_ssh.detach().cpu(),
                    tau_ssh.detach().cpu(),
                    kappa_sst.detach().cpu(),
                    m1_sst.detach().cpu(),
                    m2_sst.detach().cpu(),
                    H11_sst.detach().cpu(),
                    H12_sst.detach().cpu(),
                    H21_sst.detach().cpu(),
                    H22_sst.detach().cpu(),
                    tau_sst.detach().cpu()
                ],
                dim=1,
            ))  

        out = None
        x_simu_ssh = None
        x_simu_itrp_ssh = None
        x_simu_cond_ssh = None
        theta_ssh = None
        kappa_ssh = None
        tau_ssh = None
        m_ssh = None
        H_ssh = None
        x_simu_sst = None
        x_simu_itrp_sst = None
        x_simu_cond_sst = None
        theta_sst = None
        kappa_sst = None
        tau_sst = None
        m_sst = None
        H_sst = None

    @property
    def test_quantities(self):
        return ['inp_ssh', 'tgt_ssh', 'out_ssh',
                'inp_sst', 'tgt_sst', 'out_sst']

    @property
    def test_simu_quantities(self):
        return ['sample_x_ssh', 'mx_ssh', 'sample_xy_ssh',
                'sample_x_sst', 'mx_sst', 'sample_xy_sst',]

    @property
    def test_params_quantities(self):
        return ['kappa_ssh', 'm1_ssh', 'm2_ssh', 'H11_ssh', 'H12_ssh', 'H21_ssh', 'H22_ssh', 'tau_ssh',
                'kappa_sst', 'm1_sst', 'm2_sst', 'H11_sst', 'H12_sst', 'H21_sst', 'H22_sst', 'tau_sst']

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

        daw = 5
        rec_params_wdaw = []
        rec_simu_wdaw = []
        for i in range(daw):
            rw = np.zeros(self.rec_weight.cpu().numpy().shape)
            if self.frcst_lead is None:
                rw[i] = self.rec_weight.cpu().numpy()[2]
            else:
                rw[i] = self.rec_weight.cpu().numpy()[-1]
            # reconstruct parameters
            if isinstance(self.trainer.test_dataloaders,list):
                rec_params = self.trainer.test_dataloaders[0].dataset.reconstruct(
                        self.test_params, rw, crop = self.crop_daw
                )
            else:
                rec_params = self.trainer.test_dataloaders.dataset.reconstruct(
                            self.test_params, rw, crop = self.crop_daw
                )
            rec_params_wdaw.append(rec_params)
            
            # reconstruct simulations
            rec_da_wsimu = []
            for i in range(self.n_simu):
                if isinstance(self.trainer.test_dataloaders,list):
                    rec_simu = self.trainer.test_dataloaders[0].dataset.reconstruct(
                        [ts[:,:,:,:,:,i] for ts in self.test_simu], rw, crop = self.crop_daw
                    )
                else:
                    rec_simu = self.trainer.test_dataloaders.dataset.reconstruct(
                        [ts[:,:,:,:,:,i] for ts in self.test_simu], rw, crop = self.crop_daw
                    )
                rec_da_wsimu.append(rec_simu)
            
            if len(rec_da_wsimu)>1:
                rec_simu_wdaw.append(xr.concat(rec_da_wsimu, pd.Index(np.arange(self.n_simu), name='simu')))
            else:
                rec_simu_wdaw(rec_da_wsimu[0])

        self.test_params = xr.concat(rec_params_wdaw, pd.Index(np.arange(daw), name='daw'))
        self.test_params = self.test_params.assign_coords(
                    dict(v0=self.test_params_quantities)
                ).to_dataset(dim='v0')

        self.test_simu = xr.concat(rec_simu_wdaw, pd.Index(np.arange(daw), name='daw'))
        self.test_simu = self.test_simu.assign_coords(
                        dict(v0=self.test_simu_quantities)
                     ).to_dataset(dim='v0')

        self.test_data = xr.merge([self.test_data,self.test_simu,self.test_params])
        self.test_data = self.test_data.update({'std_simu_ssh':(('daw','time','lat','lon'),
                                        self.test_data.sample_xy_ssh.std(dim='simu').values)})
        self.test_data = self.test_data.update({'std_simu_sst':(('daw','time','lat','lon'),
                                        self.test_data.sample_xy_sst.std(dim='simu').values)})
        self.test_data = self.test_data.transpose('time', 'lat', 'lon', 'daw', 'simu')

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


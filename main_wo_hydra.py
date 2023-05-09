#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 17:26:12 2023

@author: rfablet



"""

import torch

from src.data_l63 import BaseDataModule
from src.models_l63 import Lit4dVarNet_L63
from src.models_l63 import get_constant_crop_l63
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data_l63 import create_l63_datasets

EPS_NORM_GRAD = 0. * 1.e-20  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_float32_matmul_precision('medium')
torch.set_float32_matmul_precision('high')

#cfg = get_cfg("base")
# cfg = get_cfg("xp_aug/xp_repro/quentin_repro")
cfg = OmegaConf.load('config/xp/base_l63.yaml')
print(OmegaConf.to_yaml(cfg))

dm = BaseDataModule(create_l63_datasets(cfg.datamodule.input_data.param_dataset),cfg.datamodule.param_datamodule)

mod = Lit4dVarNet_L63(cfg.model.params,patch_weight=get_constant_crop_l63(patch_dims=cfg.model.patch_weight.patch_dims,crop=cfg.model.patch_weight.crop))
#mod.load_from_checkpoint('outputs/2023-05-07/22-59-30/base_l63/checkpoints/val_mse=0.6534-epoch=379.ckpt')

ckpt = 'resL63/exp02-new/model-l63-jamesDim0_08_20unet-exp02-new-Noise02-igrad05_02-dgrad100-drop20-rnd-init00-lstm-init00-epoch=01-val_loss=4.86.ckpt'
mod.load_from_checkpoint(ckpt)

print()
print()
print(mod.hparams)
print('.................')
print(mod.model.model_VarCost.params)
#print(mod.model.model_Grad.lstm.Gates.weight)

mod.set_norm_stats = dm.norm_stats()

print('n_step = %d'%mod.model.n_step)
profiler_kwargs = {'max_epochs': 400 }

suffix_exp = 'exp%02d'%cfg.datamodule.input_data.param_dataset.flagTypeMissData + cfg.model.params.suffix_exp


filename_chkpt = 'model-l63-'+ dm.genSuffixObs        
filename_chkpt = filename_chkpt+cfg.model.params.phi_param+'-'              
filename_chkpt = filename_chkpt + suffix_exp+'-Noise%02d'%(cfg.datamodule.input_data.param_dataset.varNoise)


filename_chkpt = filename_chkpt+'-igrad%02d_%02d'%(mod.hparams.n_grad,mod.hparams.k_n_grad)+'-dgrad%d'%cfg.model.params.dim_grad_solver
filename_chkpt = filename_chkpt+'-drop%02d'%(100*cfg.model.params.dropout)
filename_chkpt = filename_chkpt+'-rnd-init%02d'%(100*mod.hparams.sig_rnd_init)
filename_chkpt = filename_chkpt+'-lstm-init%02d'%(100*mod.hparams.sig_lstm_init)

print('.... chkpt: '+filename_chkpt)
checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      dirpath= './resL63/'+suffix_exp,
                                      filename= filename_chkpt + '-{epoch:02d}-{val_loss:.2f}',
                                      save_top_k=3,
                                      mode='min')
trainer = pl.Trainer(devices=1,accelerator="gpu",  **profiler_kwargs,callbacks=[checkpoint_callback],inference_mode=False)
#trainer.fit(mod, datamodule=dm ) #dataloaders['train'], dataloaders['val'])        
#trainer.fit(mod, dataloaders['train'], dataloaders['val'])

trainer.test(mod, dataloaders=dm.test_dataloader())

print('.................')
print(mod.model.model_VarCost.params)
print()
print()

trainer.test(mod, dataloaders=dm.test_dataloader(), ckpt_path=ckpt)

print('.................')
print(mod.model.model_VarCost.params)

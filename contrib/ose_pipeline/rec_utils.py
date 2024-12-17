import pytorch_lightning
from omegaconf import OmegaConf
import hydra
import xarray as xr
import numpy as np
import pyinterp
import os
import logging
import netCDF4
import scipy.signal
from scipy import interpolate
import matplotlib.pylab as plt

def call_cfg_key(cfg, key):
    OmegaConf.set_struct(cfg, True)
    node = OmegaConf.select(cfg, key)
    return hydra.utils.call(node)


def reconstruct_from_config(config, rec_path, xp_name, data_name, best_ckpt_path):

    trainer = pytorch_lightning.Trainer(
        inference_mode= False,
        accelerator='gpu',
        devices= 1,
        logger = pytorch_lightning.loggers.CSVLogger(save_dir=rec_path, name=xp_name, version=data_name)
        )

    lit_mod = call_cfg_key(config, 'model')

    dm = call_cfg_key(config, 'datamodule')

    pytorch_lightning.seed_everything(333)
    trainer.test(
        model=lit_mod,
        datamodule=dm,
        ckpt_path=best_ckpt_path
    )
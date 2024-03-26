import operator
from pathlib import Path

import hydra
import hydra_zen
import omegaconf
import pytorch_lightning as pl
import toolz
import xarray as xr
import xrpatcher
from omegaconf import OmegaConf

import dz_lit_patch_predict





b = hydra_zen.make_custom_builds_fn()
pb = hydra_zen.make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


params = dict(
    input_path="data/prepared/gridded.nc",
    config_path="config.yaml",
    input_var="ssh",
    ckpt_path="my_checkpoint.ckpt",
    accelerator="gpu",
    strides=dict(),
    check_full_scan=True,
)

def trainer(accelerator, devices, **kwargs):
    return pl.Trainer(inference_mode=False, accelerator=accelerator, devices=devices)

def solver(config_path, ckpt_path, **kwargs):
    import torch
    model = dz_lit_patch_predict.load_from_cfg(
        config_path,
        key="model",  
    )
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    return model
    

def patcher(input_path, config_path, strides, input_var, check_full_scan, patch_dims_key="datamodule.xrds_kw.patch_dims", **kwargs):
    patches = dz_lit_patch_predict.load_from_cfg(
            cfg_path=config_path,
            key=patch_dims_key,
            call=False,
        )
    patcher = xrpatcher.XRDAPatcher(
        da=xr.open_dataset(input_path)[input_var],
        patches=patches,
        strides=strides,
        check_full_scan=check_full_scan,
    )
    return patcher

starter_predict, recipe, params = dz_lit_patch_predict.register(
    name="starter_predict",
    solver=pb(solver),
    patcher=pb(patcher),
    trainer=pb(trainer),
    params=params,
)

if __name__ == "__main__":
    starter_predict()

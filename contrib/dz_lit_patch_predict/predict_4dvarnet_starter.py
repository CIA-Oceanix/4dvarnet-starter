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


def load_from_cfg(
    cfg_path,
    key,
    overrides=None,
    overrides_targets=None,
    cfg_hydra_path=None,
    call=True,
):
    print(key)
    src_cfg = OmegaConf.load(Path(cfg_path))
    overrides = overrides or dict()
    OmegaConf.set_struct(src_cfg, True)
    if cfg_hydra_path is not None:
        hydra_cfg = OmegaConf.load(Path(cfg_hydra_path))
        OmegaConf.register_new_resolver(
            "hydra", lambda k: OmegaConf.select(hydra_cfg, k), replace=True
        )
    with omegaconf.open_dict(src_cfg):
        cfg = OmegaConf.merge(src_cfg, overrides)
    if overrides_targets is not None:
        for path, target in overrides_targets.items():
            node = OmegaConf.select(cfg, path)
            node._target_ = target
    node = OmegaConf.select(cfg, key)
    return hydra.utils.call(node) if call else node


b = hydra_zen.make_custom_builds_fn()


params = dict(
    config_path="config.yaml",
    input_var="ssh",
    ckpt_path="my_checkpoint.ckpt",
    accelerator="gpu",
    strides=dict(),
    check_full_scan=True,
)
trainer = b(pl.Trainer, inference_mode=False, accelerator="${..params.accelerator}")
solver = b(
    load_from_cfg,
    cfg_path="config.yaml",
    key="model",
    overrides=dict(
        model=dict(checkpoint_path="${.....params.ckpt_path}", map_location="cpu"),
    ),
    overrides_targets=dict(
        model="src.models.Lit4dVarNet.load_from_checkpoint",
    ),
)
patcher = b(
    xrpatcher.XRDAPatcher,
    da=b(
        toolz.pipe,
        b(xr.open_dataset, filename_or_obj="${.....input_path}"),
        b(operator.itemgetter, "${....params.input_var}"),
    ),
    patches=b(
        load_from_cfg,
        cfg_path="config.yaml",
        key="datamodule.xrds_kw.patch_dims",
        call=False,
    ),
    strides="${..params.strides}",
    check_full_scan="${..params.check_full_scan}",
)

starter_predict, predict_recipe = dz_lit_patch_predict.register(
    name="starter_predict",
    solver=solver,
    patcher=patcher,
    trainer=trainer,
    params=params,
)

if __name__ == "__main__":
    starter_predict()

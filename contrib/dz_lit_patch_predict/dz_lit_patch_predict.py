import logging
import operator
from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Optional

import hydra
import hydra_zen
import numpy as np
import omegaconf
import pytorch_lightning as pl
import toolz
import torch
import xarray as xr
import xrpatcher
from hydra.conf import HelpConf, HydraConf
from numpy.core.numeric import base_repr
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

PIPELINE_DESC = (
    "Load input observations as batches, make and store the predictions on disk"
)

## VALIDATE: Specifying input output format


def input_validation(
    input_path: str,
):  # The expected format can depend on other parameters
    """
    Take a xr.DataArray with spatial temporal observations
    Requirements:
      - input_path points to a file

    """  ## TODO: implement and document validation steps
    log.debug("Starting input validation")
    try:
        assert Path(input_path).exists(), "input_path points to a file"
        log.debug("Succesfully validated input")
    except:
        log.error("Failed to validate input, continuing anyway", exc_info=1)


def output_validation(
    output_dir: str,
):  # The expected format can depend on other parameters
    """
    Create a directory with predictions for each batch as separate netCDF4 files
    Requirements:
      - output_path points to a file
    """  ## TODO: implement and document validation steps
    log.debug("Starting output validation")
    try:
        assert Path(output_dir).exists(), "output_path points to a file"
        log.debug("Succesfully validated output")
    except:
        log.error("Failed to validate output", exc_info=1)

def load_from_cfg(
    cfg_path,
    key,
    overrides=None,
    overrides_targets=None,
    cfg_hydra_path=None,
    call=True,
):
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

PredictItem = namedtuple("PredictItem", ("input",))


class XrDataset(torch.utils.data.Dataset):
    def __init__(
        self, patcher: xrpatcher.XRDAPatcher, postpro_fns=(PredictItem._make,)
    ):
        self.patcher = patcher
        self.postpro_fns = postpro_fns or [lambda x: x.values]

    def __getitem__(self, idx):
        item = self.patcher[idx].load()
        item = toolz.thread_first(item, *self.postpro_fns)
        return item

    def __len__(self):
        return len(self.patcher)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


class LitModel(pl.LightningModule):
    def __init__(
        self,
        patcher,
        model,
        norm_stats,
        save_dir="batch_preds",
        out_dims=("time", "lat", "lon"),
    ):
        super().__init__()
        self.patcher = patcher
        self.solver = model
        self.norm_stats = norm_stats
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.out_dims = out_dims
        self.bs = 0

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        outputs = self.solver(batch)

        self.bs = self.bs or outputs.shape[0]
        m, s = self.norm_stats
        outputs = (outputs * s + m).cpu().numpy()
        numitem = outputs.shape[0]
        num_devices = self.trainer.num_devices * self.trainer.num_nodes
        item_idxes = (
            batch_idx * self.bs + torch.arange(numitem)
        ) * num_devices + self.global_rank
        assert len(self.out_dims) == len(outputs[0].shape)
        for i, idx in enumerate(item_idxes):
            out = outputs[i]
            c = self.patcher[idx].coords.to_dataset()[list(self.out_dims)]
            da = xr.DataArray(out, dims=self.out_dims, coords=c.coords)
            da.astype(np.float32).to_netcdf(self.save_dir / f"{idx}.nc")


## PROCESS: Parameterize and implement how to go from input_files to output_files
def run(
    input_path: str = "???",
    output_dir: str = "???",
    norm_stats: list = None,
    dl_kws: dict = dict(batch_size=4, num_workers=1),
    trainer_fn="???",
    patcher_fn="???",
    solver_fn="???",
    params: Optional[dict] = None,
    _skip_val: bool = False,
):
    print(patcher_fn)
    log.info("Starting")
    if not _skip_val:
        input_validation(input_path=input_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)  # Make output directory

    log.info("Instantiating Trainer")
    trainer = trainer_fn(**params)
    log.info("Instantiating Patcher")
    patcher = patcher_fn(**params)
    log.info("Instantiating Solver")
    solver = solver_fn(**params)

    if norm_stats is None:
        norm_stats = patcher.da.mean().item(), patcher.da.std().item()
    mean, std = norm_stats
    logging.info(f"{norm_stats=}")
    torch_ds = XrDataset(
        patcher=patcher,
        postpro_fns=(
            lambda item: PredictItem._make((item.values.astype(np.float32),)),
            lambda item: item._replace(input=(item.input - mean) / std),
        ),
    )

    dl = torch.utils.data.DataLoader(torch_ds, **dl_kws)
    log.info(f"{next(iter(dl)).input.shape=}")
    litmod = LitModel(patcher, solver, norm_stats, save_dir=output_dir)
    trainer.predict(litmod, dataloaders=dl)

    if not _skip_val:
        output_validation(output_dir=output_dir)


## EXPOSE: document, and configure CLI
run.__doc__ = f"""
Pipeline description:
    {PIPELINE_DESC}

Input description:
    {input_validation.__doc__}

Output description:
    {output_validation.__doc__}

Returns:
    None
"""


def register(name, solver, patcher, trainer, params=None):
    if isinstance(params, dict):
        params = hydra_zen.make_config(**params)

    store = hydra_zen.store(group="patch_predict", package="_global_")

    params_store = hydra_zen.store(group="patch_predict/params", package="params")
    patcher_store = store(group="patch_predict/patcher", package="patcher_fn")
    trainer_store = store(group="patch_predict/trainer", package="trainer_fn")
    solver_store = store(group="patch_predict/solver", package="solver_fn")

    params_store(params, name=name)
    patcher_store(patcher, name=name)
    trainer_store(trainer, name=name)
    solver_store(solver, name=name)

    store = hydra_zen.store()
    store(HydraConf(help=HelpConf(header=run.__doc__, app_name=__name__)))

    base_config = hydra_zen.builds(
        run,
        populate_full_signature=True,
        zen_partial=True,
        zen_dataclass=dict(cls_name="BasePredict"),
    )

    _recipe = hydra_zen.make_config(
            bases=(base_config,),
            hydra_defaults=[
                "_self_",
                {"/patch_predict/patcher": name},
                {"/patch_predict/trainer": name},
                {"/patch_predict/solver": name},
                {"/patch_predict/params": name},
            ],
    )

    store(
        _recipe,
        name=name,
        group="patch_predict",
        package="_global_",
    )
    # Create a  partial configuration associated with the above function (for easy extensibility)

    store.add_to_hydra_store(overwrite_ok=True)
    patcher_store.add_to_hydra_store(overwrite_ok=True)
    trainer_store.add_to_hydra_store(overwrite_ok=True)
    solver_store.add_to_hydra_store(overwrite_ok=True)
    # Create CLI endpoint


    with hydra.initialize(version_base='1.3', config_path='.'):
        cfg = hydra.compose("/patch_predict/" + name)
    from omegaconf import OmegaConf
    recipe = hydra_zen.make_config(
        **{k: node for k,node in OmegaConf.to_container(cfg).items()
            if k not in ('_target_', '_partial_', '_args_', '_convert_', '_recursive_')},
        bases=(base_config,),
        zen_dataclass={'cls_name': f'{"".join(x.capitalize() for x in name.lower().split("_"))}Recipe'}
    )
    zen_endpoint = hydra_zen.zen(run)
    api_endpoint = hydra.main(
        config_name="patch_predict/" + name, version_base="1.3", config_path="."
    )(zen_endpoint)

    return api_endpoint, recipe, params


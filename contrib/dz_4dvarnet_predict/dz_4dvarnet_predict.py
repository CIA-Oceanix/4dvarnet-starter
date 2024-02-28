from collections import namedtuple
import logging
import hydra_zen
import hydra
from hydra.conf import HydraConf, HelpConf
from pathlib import Path
from omegaconf import OmegaConf
import xarray as xr
import torch
import toolz
import pytorch_lightning as pl
import xrpatcher
import numpy as np

log = logging.getLogger(__name__)

PIPELINE_DESC = "Load input observations as batches, make and store the predictions on disk"

## VALIDATE: Specifying input output format

def input_validation(input_path: str): # The expected format can depend on other parameters
    """
    Take a xr.DataArray with spatial temporal observations
    Requirements:
      - input_path points to a file

    """ ## TODO: implement and document validation steps
    log.debug('Starting input validation')
    try:
        assert Path(input_path).exists(), "input_path points to a file"
        log.debug('Succesfully validated input')
    except:
        log.error('Failed to validate input, continuing anyway', exc_info=1)

def output_validation(output_dir: str): # The expected format can depend on other parameters
    """
    Create a directory with predictions for each batch as separate netCDF4 files
    Requirements:
      - output_path points to a file
    """ ## TODO: implement and document validation steps
    log.debug('Starting output validation')
    try:
        assert Path(output_dir).exists(), "output_path points to a file"
        log.debug('Succesfully validated output')
    except:
        log.error('Failed to validate output', exc_info=1)

PredictItem = namedtuple('PredictItem', ('input',))

class XrDataset(torch.utils.data.Dataset):
    def __init__(self, patcher: xrpatcher.XRDAPatcher, postpro_fns=(PredictItem._make,)):
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
    def __init__(self, patcher, model, norm_stats, save_dir='batch_preds', out_dims=('time', 'lat', 'lon')):
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
        item_idxes = (batch_idx * self.bs + torch.arange(numitem)
        )*num_devices + self.global_rank
        for i, idx in enumerate(item_idxes):
            out = outputs[i]
            c = self.patcher[idx].coords.to_dataset()[list(self.out_dims)]
            da = xr.DataArray(out, dims=self.out_dims, coords=c.coords)
            da.to_netcdf(self.save_dir / f"{idx}.nc")



## PROCESS: Parameterize and implement how to go from input_files to output_files
def run(
    input_path: str = '???',
    output_dir: str = '???',
    config_path: str = '???',
    trainer_config_key: str = 'trainer',
    model_config_key: str = 'model',
    patcher_config_key: str = 'patcher',
    model_config_path: str = '${.config_path}',
    trainer_config_path: str = '${.config_path}',
    patcher_config_path: str = '${.config_path}',
    weight_path: str = '???',
    norm_stats: list = None,
    batch_size: int = 4,
    _skip_val: bool = False,
):
    log.info("Starting")
    if not _skip_val:
      input_validation(input_path=input_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True) # Make output directory

    # Load model and patcher config (with overrides)
    model_cfg = OmegaConf.load(model_config_path)[model_config_key]
    model = hydra.utils.call(model_cfg)

    patcher_cfg = OmegaConf.load(patcher_config_path)[patcher_config_key]

    da=xr.open_dataarray(input_path)
    patcher: xrpatcher.XRDAPatcher = hydra.utils.call(
        patcher_cfg,
        da=da,
    )

    if norm_stats is None:
        norm_stats = da.mean().item(), da.std().item()
    mean, std = norm_stats
    logging.info(F"{norm_stats=}")
    torch_ds = XrDataset(
        patcher=patcher,
        postpro_fns=(
            lambda item: PredictItem._make((item.values.astype(np.float32),)),
            lambda item: item._replace(input=(item.input - mean) / std),
        )
    )

    dl = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=batch_size
    )

    litmod = LitModel(
        patcher, model.solver, norm_stats, save_dir=output_dir
    )

    trainer_cfg = OmegaConf.load(trainer_config_path)[trainer_config_key]
    trainer: pl.Trainer = hydra.utils.call(trainer_cfg)

    trainer.predict(litmod, dataloaders=dl, ckpt_path=weight_path)

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
# Create a configuration associated with the above function (cf next cell)
main_config =  hydra_zen.builds(run, populate_full_signature=True)

cfg = main_config(
    input_path='/home/administrateur/.datasets/example_ocb_input.nc',
    output_dir='./outputs/',
    patcher_config_path='/home/administrateur/Lab/archives/starter/base/base-bigger-model-1000-epo/.hydra/overrides.yaml',
    config_path='/home/administrateur/Lab/archives/starter/base/base-bigger-model-1000-epo/.hydra/config.yaml',
    trainer_config_path='/home/administrateur/Lab/archives/starter/base/base-bigger-model-1000-epo/.hydra/overrides.yaml',
    weight_path='/home/administrateur/Lab/archives/starter/base/base-bigger-model-1000-epo/base/checkpoints/val_mse=1.97545-epoch=893.ckpt',
    norm_stats=[0.3174592795757612, 0.3886552185297686],
)

# Wrap the function to accept the configuration as input
zen_endpoint = hydra_zen.zen(run)

#Store the config
store = hydra_zen.ZenStore()
store(HydraConf(help=HelpConf(header=run.__doc__, app_name=__name__)))
store(main_config, name=__name__)
store(cfg, name='test')

store.add_to_hydra_store(overwrite_ok=True)

# Create CLI endpoint
api_endpoint = hydra.main(
    # config_name=__name__, version_base="1.3", config_path=None
    config_name='test', version_base="1.3", config_path=None
)(zen_endpoint)


if __name__ == '__main__':

    api_endpoint()
    # cfg = main_config(
    #     input_path='/home/administrateur/.datasets/example_ocb_input.nc',
    #     output_dir='./outputs/',
    #     patcher_config_path='/home/administrateur/Lab/archives/starter/base/base-bigger-model-1000-epo/.hydra/overrides.yaml',
    #     config_path='/home/administrateur/Lab/archives/starter/base/base-bigger-model-1000-epo/.hydra/config.yaml',
    #     trainer_config_path='/home/administrateur/Lab/archives/starter/base/base-bigger-model-1000-epo/.hydra/overrides.yaml',
    #     weight_path='/home/administrateur/Lab/archives/starter/base/base-bigger-model-1000-epo/base/checkpoints/val_mse=1.97545-epoch=893.ckpt',
    #     norm_stats=[0.3174592795757612, 0.3886552185297686],
    # )

    # zen_endpoint(cfg)
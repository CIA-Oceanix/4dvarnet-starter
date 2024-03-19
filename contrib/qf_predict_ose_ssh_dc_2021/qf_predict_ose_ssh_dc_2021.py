import logging
from pathlib import Path

import dc_ose_2021
import dz_lit_patch_predict
import hydra
import hydra_zen
import qf_pipeline
from hydra.conf import HelpConf, HydraConf

log = logging.getLogger(__name__)

PIPELINE_DESC = "Inference of the 2021 data challenge on ose ssh"

params = dict(dc_data_params=dc_ose_2021)
stages = {}

pipeline = qf_pipeline.register(
    name="dc_ose_2021_inference", stages=stages, params=params
)
if __name__ == "__main__":
    pipeline()

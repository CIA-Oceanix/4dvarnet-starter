import torch
import os
import random
torch.set_float32_matmul_precision('high')
from pytorch_lightning import loggers

def base_test(trainer, dm, lit_mod, ckpt_path):

    '''
    ckpt = torch.load(ckpt_path)["state_dict"]
    lit_mod.load_state_dict(ckpt)

    dm.setup()
    lit_mod._norm_stats = dm.norm_stats()
    dm._norm_stats = dm.norm_stats()
    '''

    version = 'version_' + str(random.randint(0, 100000))
    save_dir = "/DATASET/mbeauchamp/DMI/results"
    logger_name = "lightning_logs"
    print(os.path.join(save_dir, logger_name, version))
    tb_logger = loggers.TensorBoardLogger(save_dir=save_dir,
                                                   name=logger_name,
                                                   version=version)
    trainer.logger = tb_logger

    trainer.test(lit_mod, datamodule=dm, ckpt_path=ckpt_path)

import json
import os

import torch
torch.set_float32_matmul_precision('high')


def save_norm_stats(dm, trainer):
    """Persist the (mean, std) used to normalize tgt/input during training next to the
    checkpoints, so OSE inference can reuse them instead of recomputing stats from
    whatever short time slice happens to be passed as the inference 'train' domain."""
    dirpath = trainer.checkpoint_callback.dirpath
    os.makedirs(dirpath, exist_ok=True)
    mean, std = dm.norm_stats()
    with open(os.path.join(dirpath, 'norm_stats.json'), 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)


def base_training(trainer, dm, lit_mod, ckpt=None):
    print('START HERE')
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    save_norm_stats(dm, trainer)
    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')


def multi_dm_training(trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    if test_fn is not None:
        if test_dm is None:
            test_dm = dm
        lit_mod._norm_stats = test_dm.norm_stats()

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.callbacks = []
        trainer.test(lit_mod, datamodule=test_dm, ckpt_path=best_ckpt_path)

        print("\nBest ckpt score:")
        print(test_fn(lit_mod).to_markdown())
        print("\n###############")

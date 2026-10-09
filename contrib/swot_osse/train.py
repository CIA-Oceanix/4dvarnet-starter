"""Fine-tuning entry point: start from the weights of an existing checkpoint, with a fresh optimiser and schedule.

src.train.base_training passes the checkpoint to trainer.fit(ckpt_path=...), which resumes the whole training state
(epoch counter, optimiser, learning-rate schedule). For fine-tuning on a new input distribution we only want the
weights.
"""
import torch


def finetune_from_weights(trainer, dm, lit_mod, init_ckpt):
    state = torch.load(init_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    result = lit_mod.load_state_dict(state, strict=False)
    print("initial weights from", init_ckpt, "| missing:", result.missing_keys, "| unexpected:", result.unexpected_keys)
    if trainer.logger is not None:
        print("Logdir:", trainer.logger.log_dir)
    trainer.fit(lit_mod, datamodule=dm)

import torch
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    # ckpt = '/homes/g24meda/lab/4dvarnet-starter/outputs/QG_and_bilin_whole_lrgmod_01_lrgrad_100_nstep_20_sigma2_kernelsize21_alpha1_1_alpha2_05_avgpool2_dt10min_aug3/16-49-26/QG_and_bilin_whole_4_nadirs_DC_2020a_ssh/checkpoints/val_mse=17.05650-epoch=001.ckpt'
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')
    #trainer.test(lit_mod, datamodule=dm, ckpt_path='/homes/g24meda/lab/4dvarnet-starter/outputs/QG_and_bilin_lrgmod_01_lrgrad_100_nstep_20_sigma2_kernelsize21_alpha1_1_alpha2_05_avgpool2_dt10min/15-50-08/QG_and_bilin_4_nadirs_DC_2020a_ssh/checkpoints/val_mse=19.49099-epoch=050.ckpt')

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

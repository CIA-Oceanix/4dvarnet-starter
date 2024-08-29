import torch
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    #ckpt = '/homes/g24meda/lab/4dvarnet-starter/outputs/all_year_Id_lrgmod_01_lrgrad_100_nstep_20_sigma2_kernelsize21_alpha1_1_alpha2_05_avgpool2_dimhidd48_augdata1/17-27-06/Id_whole_4_nadirs_DC_2020a_ssh/checkpoints/val_mse=7.06618-epoch=010.ckpt'
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    trainer.test(lit_mod, datamodule=dm, ckpt_path='best')
    #trainer.test(lit_mod, datamodule=dm, ckpt_path='/homes/g24meda/lab/4dvarnet-starter/outputs/Id_lrgmod_01_lrgrad_100_nstep_20_sigma2_kernelsize21_alpha1_1_alpha2_05_avgpool2_dt10min/11-28-12/Id_4_nadirs_DC_2020a_ssh/checkpoints/val_mse=7.58272-epoch=094.ckpt')

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

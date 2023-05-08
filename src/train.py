import torch
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    #trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    
    lit_mod.set_norm_stats = dm.norm_stats()
    lit_mod.load_from_checkpoint('outputs/2023-05-07/22-59-30/base_l63/checkpoints/val_mse=0.6534-epoch=379.ckpt')
    
    print()
    print('.......................')
    print(lit_mod.model.modl_Grad.lstm.Gates.weight[0,0,:,:])
    print()
    
    trainer.test(lit_mod, dataloaders=dm.val_dataloader(), ckpt_path=ckpt)
    
    print(lit_mod.model.modl_Grad.lstm.Gates.weight[0,0,:,:])
    
    trainer.test(lit_mod, dataloaders=dm.test_dataloader())#, ckpt_path=ckpt)
    
    if test_fn is not None:
        if test_dm is None:
            test_dm = dm
        #lit_mod.norm_stats = test_dm.norm_stats()

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.callbacks = []
        trainer.test(lit_mod, datamodule=test_dm, ckpt_path=best_ckpt_path)

        print("\nBest ckpt score:")
        print(test_fn(lit_mod).to_markdown())
        print("\n###############")

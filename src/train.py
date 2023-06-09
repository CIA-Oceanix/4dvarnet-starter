import torch
torch.set_float32_matmul_precision('high')

def base_training(trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    lit_mod.set_norm_stats = dm.norm_stats()
    
    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

            
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

def fine_tuning(trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None,update_params=False):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    if update_params == True :
        if ckpt is not None:
            print('.... Load model: '+ckpt)
            lit_mod.load_state_dict(torch.load(ckpt)['state_dict'])
    
        cfg_params = lit_mod.hparams
        
        print('')
        lit_mod = lit_mod.load_from_checkpoint(ckpt)
        print('...... cfg parameters from chekpoint',flush=True)
        print(lit_mod.hparams)
           
        # force optimization parameters
        lit_mod.update_params( n_grad = cfg_params.n_grad , k_n_grad = cfg_params.k_n_grad, 
                              lr_grad = cfg_params.lr_grad, lr_rnd = cfg_params.lr_rnd,
                              sig_rnd_init = cfg_params.sig_rnd_init, sig_lstm_init = cfg_params.sig_lstm_init,
                              param_lstm_step = cfg_params.param_lstm_step,
                              sig_obs_noise = cfg_params.sig_obs_noise,
                              post_projection = cfg_params.post_projection,
                              post_median_filter = cfg_params.post_median_filter,
                              median_filter_width = cfg_params.median_filter_width)

        print('...... Updated parameters from cfg files')
        print(lit_mod.hparams)
    
        # normalisation parameters
        lit_mod.set_norm_stats = dm.norm_stats()
    
        trainer.fit(lit_mod, datamodule=dm)
    else:
        trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
            
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

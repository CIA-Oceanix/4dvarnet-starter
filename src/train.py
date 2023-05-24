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

def fine_tuning(trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    if ckpt is not None:
        print('.... Load model: '+ckpt)
        lit_mod.load_state_dict(torch.load(ckpt)['state_dict'])

    cfg_params = lit_mod.hparams
    
    print('')
    lit_mod = lit_mod.load_from_checkpoint(ckpt)
    print('...... cfg parameters from chekpoint',flush=True)
    print(lit_mod.hparams)
       
    # force optimization parameters
    lit_mod.hparams.n_grad = cfg_params.n_grad
    lit_mod.hparams.k_n_grad = cfg_params.k_n_grad
    
    lit_mod.hparams.lr_grad = cfg_params.lr_grad
    lit_mod.hparams.lr_rnd = cfg_params.lr_rnd
    lit_mod.hparams.sig_rnd_init = cfg_params.sig_rnd_init
    lit_mod.hparams.sig_lstm_init = cfg_params.sig_lstm_init
    lit_mod.hparams.param_lstm_step = cfg_params.param_lstm_step

    # force training parameters
    lit_mod.hparams.alpha_prior = cfg_params.alpha_prior
    lit_mod.hparams.alpha_mse = cfg_params.alpha_mse
    lit_mod.hparams.alpha_var_cost_grad = cfg_params.alpha_var_cost_grad
    lit_mod.hparams.lr_grad = cfg_params.lr_grad
    lit_mod.hparams.lr_rnd = cfg_params.lr_rnd
    lit_mod.hparams.sig_rnd_init = cfg_params.sig_rnd_init
    lit_mod.hparams.sig_lstm_init = cfg_params.sig_lstm_init
    lit_mod.hparams.degradation_operator = cfg_params.degradation_operator
    lit_mod.hparams.sig_perturbation_grad = cfg_params.sig_perturbation_grad
    lit_mod.hparams.alpha_perturbation_grad = cfg_params.alpha_perturbation_grad
    lit_mod.hparams.gamma_degradation = cfg_params.gamma_degradation
    lit_mod.hparams.param_lstm_step = cfg_params.param_lstm_step

    print('...... Updated parameters from cfg files')
    print(lit_mod.hparams)
    
    # normalisation parameters
    lit_mod.set_norm_stats = dm.norm_stats()

    trainer.fit(lit_mod, datamodule=dm)

            
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

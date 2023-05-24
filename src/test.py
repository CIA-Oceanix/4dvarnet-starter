import torch
import numpy as np

torch.set_float32_matmul_precision('high')

def base_testing(trainer, dm, lit_mod,ckpt):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()


    #lit_mod.set_norm_stats = dm.norm_stats()
    #lit_mod.load_from_checkpoint(ckpt)
    
    # validation dataset
    trainer.callbacks = []
    lit_mod.set_norm_stats = dm.norm_stats()
    
    
    # load checkpoints
    cfg_params = lit_mod.hparams
    
    print('')
    #print('...... Loaded cfg parameters')
    #print(cfg_params)
    lit_mod = lit_mod.load_from_checkpoint(ckpt)
    print('...... cfg parameters from chekpoint',flush=True)
    print(lit_mod.hparams)
       
    lit_mod.set_norm_stats = dm.norm_stats()
    
    # force optimization parameters
    lit_mod.hparams.n_grad = cfg_params.n_grad
    lit_mod.hparams.k_n_grad = cfg_params.k_n_grad
    
    lit_mod.hparams.lr_grad = cfg_params.lr_grad
    lit_mod.hparams.lr_rnd = cfg_params.lr_rnd
    lit_mod.hparams.sig_rnd_init = cfg_params.sig_rnd_init
    lit_mod.hparams.sig_lstm_init = cfg_params.sig_lstm_init
    #lit_mod.hparams.param_lstm_step = cfg_params.param_lstm_step
    
    print('...... Updated parameters from cfg files')
    print(lit_mod.hparams)
 
    print('............... Model evaluation on validation dataset')
    trainer.test(lit_mod, dataloaders=dm.val_dataloader())
    
    #trainer.test(lit_mod, dataloaders=dm.val_dataloader(),ckpt_path=ckpt)

    X_train, x_train, mask_train, x_train_Init, x_train_obs = dm.input_data[0]    
    idx_val = X_train.shape[0]-500
    
    X_val = X_train[idx_val::,:,:]
    mask_val = mask_train[idx_val::,:,:,:].squeeze()
    var_val  = np.mean( (X_val - np.mean(X_val,axis=0))**2 )
    mse = np.mean( (lit_mod.x_rec-X_val) **2 ) 
    mse_i   = np.mean( (1.-mask_val.squeeze()) * (lit_mod.x_rec-X_val) **2 ) / np.mean( (1.-mask_val) )
    mse_r   = np.mean( mask_val.squeeze() * (lit_mod.x_rec-X_val) **2 ) / np.mean( mask_val )
    
    nmse = mse / var_val
    nmse_i = mse_i / var_val
    nmse_r = mse_r / var_val
    
    print("..... Assimilation performance (validation data)")
    print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
    print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
    print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))


    # test dataset
    print()
    print()
    print('............... Model evaluation on test dataset')
    trainer.test(lit_mod, dataloaders=dm.test_dataloader())
    X_test, x_test, mask_test, x_test_Init, x_test_obs = dm.input_data[1]

    var_test  = np.mean( (X_test - np.mean(X_test,axis=0))**2 )
    mse = np.mean( (lit_mod.x_rec-X_test) **2 ) 
    mse_i   = np.mean( (1.-mask_test.squeeze()) * (lit_mod.x_rec-X_test) **2 ) / np.mean( (1.-mask_test) )
    mse_r   = np.mean( mask_test.squeeze() * (lit_mod.x_rec-X_test) **2 ) / np.mean( mask_test )
    
    nmse = mse / var_test
    nmse_i = mse_i / var_test
    nmse_r = mse_r / var_test
    
    print("..... Assimilation performance (test data)")
    print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
    print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
    print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))     
    
    print()
    print()
    print('............... Second run on test dataset to check stochasticity')
    x_rec_1 = 1. * lit_mod.x_rec
    trainer.test(lit_mod, dataloaders=dm.test_dataloader())#, ckpt_path=ckpt)
    var_rec = np.mean( (x_rec_1-lit_mod.x_rec)**2 )
    bias_rec = np.mean( (x_rec_1-lit_mod.x_rec) )
    print('..')
    print('.. Mean difference between 2 runs : %.3f'%bias_rec)
    print('.. MSE between 2 runs             : %.3f'%var_rec)

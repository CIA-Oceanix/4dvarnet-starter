import torch
import numpy as np

torch.set_float32_matmul_precision('high')

def base_testing(trainer, dm, lit_mod,chkpt):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    #lit_mod.set_norm_stats = dm.norm_stats()
    #lit_mod.load_from_checkpoint(ckpt)
    
    # validation dataset
    trainer.callbacks = []
    lit_mod.set_norm_stats = dm.norm_stats()
    
    trainer.test(lit_mod, dataloaders=dm.val_dataloader(),ckpt_path=chkpt)

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
    trainer.test(lit_mod, dataloaders=dm.test_dataloader())#, ckpt_path=ckpt)

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
    
    x_rec_1 = 1. * lit_mod.x_rec
    trainer.test(lit_mod, dataloaders=dm.test_dataloader())#, ckpt_path=ckpt)
    var_rec = np.mean( (x_rec_1-lit_mod.x_rec)**2 )
    bias_rec = np.mean( (x_rec_1-lit_mod.x_rec) )
    print('..')
    print('.. Mean difference between 2 runs : %.3f'%bias_rec)
    print('.. MSE between 2 runs             : %.3f'%var_rec)

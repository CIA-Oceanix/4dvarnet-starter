import torch
import numpy as np
import xarray as xr

torch.set_float32_matmul_precision('high')


def save_netcdf(saved_path1, gt, pred, obs,mask):
    '''
    saved_path1: string
    pred: 3d numpy array (4DVarNet-based predictions)
    lon: 1d numpy array
    lat: 1d numpy array
    time: 1d array-like of time corresponding to the experiment
    '''
    xrdata = xr.Dataset( \
        data_vars={'gt': (('idx', 'l63','time'), gt),
                   'obs': (('idx', 'l63','time'), obs),
                   'mask': (('idx', 'l63','time'), mask),
                   'rec': (('idx', 'l63','time','members'), pred)}, \
        coords={'idx': np.arange(gt.shape[0]), 'l63': np.arange(gt.shape[1]), 'time': np.arange(gt.shape[2]), 'members': np.arange(pred.shape[3])})
    xrdata.to_netcdf(path=saved_path1, mode='w')


def base_testing(trainer, dm, lit_mod,ckpt,num_members=1):
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
    m_NormObs = lit_mod.model.model_VarCost.normObs
    m_NormPhi = lit_mod.model.model_VarCost.normPrior
    
    print('')
    print('...... Loaded model: '+ckpt)
    #print('...... Loaded cfg parameters')
    #print(cfg_params)

    print(lit_mod.model.mod_H,flush=True)
    
    print('............... Model evaluation on validation dataset')
    #trainer.test(lit_mod, dataloaders=dm.val_dataloader(), ckpt_path=ckpt)
    lit_mod = lit_mod.load_from_checkpoint(ckpt)

    print('...... cfg parameters from chekpoint',flush=True)
    print(lit_mod.hparams)
       
    lit_mod.set_norm_stats = dm.norm_stats()
    
    # force optimization parameters
    lit_mod.update_params( n_grad = cfg_params.n_grad , k_n_grad = cfg_params.k_n_grad, 
                          lr_grad = cfg_params.lr_grad, lr_rnd = cfg_params.lr_rnd,
                          sig_rnd_init = cfg_params.sig_rnd_init, sig_lstm_init = cfg_params.sig_lstm_init,
                          param_lstm_step = cfg_params.param_lstm_step,
                          sig_obs_noise = cfg_params.sig_obs_noise,
                          post_projection = cfg_params.post_projection,
                          post_median_filter = cfg_params.post_median_filter,
                          median_filter_width = cfg_params.median_filter_width)
    
    # update normObs
    lit_mod.model.model_VarCost.normObs = m_NormObs
    
    
    print('...... Updated parameters from cfg files')
    print(lit_mod.hparams)
    print('.... param_lstm_step = %d'%lit_mod.model.param_lstm_step)
    print()
 
    print('............... Model evaluation on validation dataset')
    trainer.test(lit_mod, dataloaders=dm.val_dataloader())
    
    #trainer.test(lit_mod, dataloaders=dm.val_dataloader(),ckpt_path=ckpt)

    X_train, x_train, mask_train, x_train_Init, x_train_obs = dm.input_data[0]    
    idx_val = X_train.shape[0]-500
    
    X_val = X_train[idx_val::,:,:]
    mask_val = mask_train[idx_val::,:,:,:].squeeze()
    
    X_val = X_val[:,:,cfg_params.dt_mse_test:x_train.shape[2]-cfg_params.dt_mse_test]
    mask_val = mask_val[:,:,cfg_params.dt_mse_test:x_train.shape[2]-cfg_params.dt_mse_test]
    x_rec = lit_mod.x_rec[:,:,cfg_params.dt_mse_test:x_train.shape[2]-cfg_params.dt_mse_test]
    
    var_val  = np.mean( (X_val - np.mean(X_val,axis=0))**2 )
    mse = np.mean( (x_rec-X_val) **2 ) 
    mse_i   = np.mean( (1.-mask_val.squeeze()) * (x_rec-X_val) **2 ) / np.mean( (1.-mask_val) )
    mse_r   = np.mean( mask_val.squeeze() * (x_rec-X_val) **2 ) / np.mean( mask_val )
    
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

    X_test = X_test[:,:,cfg_params.dt_mse_test:x_train.shape[2]-cfg_params.dt_mse_test]
    mask_test = mask_test[:,:,cfg_params.dt_mse_test:x_train.shape[2]-cfg_params.dt_mse_test]
    x_rec = lit_mod.x_rec[:,:,cfg_params.dt_mse_test:x_train.shape[2]-cfg_params.dt_mse_test]

    var_test  = np.mean( (X_test - np.mean(X_test,axis=0))**2 )
    mse = np.mean( (x_rec-X_test) **2 ) 
    mse_i   = np.mean( (1.-mask_test.squeeze()) * (x_rec-X_test) **2 ) / np.mean( (1.-mask_test) )
    mse_r   = np.mean( mask_test.squeeze() * (x_rec-X_test) **2 ) / np.mean( mask_test )
    
    nmse = mse / var_test
    nmse_i = mse_i / var_test
    nmse_r = mse_r / var_test
    
    print("..... Assimilation performance (test data)")
    print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
    print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
    print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))     
    
    print()
    print()
    x_rec = np.reshape(x_rec,(x_rec.shape[0],x_rec.shape[1],x_rec.shape[2],1))
    if num_members > 1 :
        for _ii in range(1,num_members):
        
            print('............... run %d on test dataset to generate members'%_ii)
            trainer.test(lit_mod, dataloaders=dm.test_dataloader())#, ckpt_path=ckpt)
            x_rec_ii = lit_mod.x_rec[:,:,cfg_params.dt_mse_test:x_train.shape[2]-cfg_params.dt_mse_test]
            x_rec_ii = np.reshape(x_rec_ii,(x_rec_ii.shape[0],x_rec_ii.shape[1],x_rec_ii.shape[2],1))

            x_rec = np.concatenate((x_rec,x_rec_ii),axis=3)
     
    mean_x_rec = np.mean( x_rec , axis = 3)
    mean_x_rec = np.reshape(mean_x_rec,(mean_x_rec.shape[0],mean_x_rec.shape[1],mean_x_rec.shape[2],1))
    
    var_rec = np.mean( (x_rec-mean_x_rec)**2 )
    max_diff = np.max( np.abs(x_rec-mean_x_rec) )
    
    
    # metrics for the mean among members
    mean_x_rec = mean_x_rec.squeeze()
    mse = np.mean( (mean_x_rec-X_test) **2 ) 
    
    nmse = mse / var_test
    
    print()
    print('.. Metrics for mean member')
    print(".. MSE mean member (test data): %.3f / %.3f"%(mse,nmse))
    print('.. Variance among members runs             : %.3f'%var_rec)
    print('.. Maximum absolute difference between 2 runs : %.3f'%max_diff)

    median_x_rec = np.median(x_rec , axis = 3)
    mse = np.mean( (median_x_rec-X_test) **2 )     
    nmse = mse / var_test
    median_x_rec = np.reshape(median_x_rec,(median_x_rec.shape[0],median_x_rec.shape[1],median_x_rec.shape[2],1))
    var_rec = np.mean( (x_rec-median_x_rec)**2 )
    max_diff = np.max( np.abs(x_rec-median_x_rec) )
    
    print()
    print('.. Metrics for median member')
    print(".. MSE median member (test data): %.3f / %.3f"%(mse,nmse))
    print('.. Variance among members runs             : %.3f'%var_rec)
    print('.. Maximum absolute difference between 2 runs : %.3f'%max_diff)
    
    
    # saving dataset
    result_path = ckpt.replace('.ckpt','_res.nc')
    x_test_obs = lit_mod.x_obs[:,:,cfg_params.dt_mse_test:x_train.shape[2]-cfg_params.dt_mse_test]
    print('..... save .c file with results: '+result_path)
    save_netcdf(result_path, X_test, x_rec, x_test_obs.squeeze(), mask_test.squeeze() )
    

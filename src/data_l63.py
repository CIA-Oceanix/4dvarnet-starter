#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytorch_lightning as pl
import numpy as np
import scipy
import torch.utils.data
import xarray as xr
#import itertools
#import functools as ft
#import tqdm
#from collections import namedtuple

from netCDF4 import Dataset
#from sklearn import decomposition
from scipy.integrate import solve_ivp
from sklearn.feature_extraction import image

def AnDA_Lorenz_63(S,t,sigma,rho,beta):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(S[1]-S[0]);
    x_2 = S[0]*(rho-S[2])-S[1];
    x_3 = S[0]*S[1] - beta*S[2];
    dS  = np.array([x_1,x_2,x_3]);
    return dS


class time_series:
  values = 0.
  time   = 0.


def create_l63_datasets(param_dataset): 
    rateMissingData = (1-1./param_dataset.sampling_step)
    sigNoise = np.sqrt( param_dataset.varNoise )
    genSuffixObs = param_dataset.genSuffixObs
    
    if param_dataset.flag_load_all_data == False :
     
        #data_module.flag_generate_L63_data = False    
        if param_dataset.flag_generate_L63_data :
            ## data generation: L63 series

            class GD:
                model = 'Lorenz_63'
                class parameters:
                    sigma = 10.0
                    rho   = 28.0
                    beta  = 8.0/3
                dt_integration = 0.01 # integration time
                dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
                #dt_obs = 8 # number of integration times between consecutive observations (for yo)
                #var_obs = np.array([0,1,2]) # indices of the observed variables
                nb_loop_train = 10**2 # size of the catalog
                nb_loop_test = 20000 # size of the true state and noisy observations
                #sigma2_catalog = 0.0 # variance of the model error to generate the catalog
                #sigma2_obs = 2.0 # variance of the observation error to generate observation

            GD = GD()    
            y0 = np.array([8.0,0.0,30.0])
            tt = np.arange(GD.dt_integration,GD.nb_loop_test*GD.dt_integration+0.000001,GD.dt_integration)
            #S = odeint(AnDA_Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
            S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[0.,5+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=np.arange(0,5+0.000001,GD.dt_integration),method='RK45')
            
            y0 = S.y[:,-1];
            S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[GD.dt_integration,GD.nb_loop_test+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=tt,method='RK45')
            S = S.y.transpose()
            
            
            ####################################################
            ## Generation of training and test dataset
            ## Extraction of time series of dT time steps            
              
            xt = time_series()
            xt.values = S
            xt.time   = tt
            # extract subsequences
            dataTrainingNoNaN = image.extract_patches_2d(xt.values[0:12000:param_dataset.time_step,:],(param_dataset.dT,3),max_patches=param_dataset.NbTraining)
            dataTestNoNaN     = image.extract_patches_2d(xt.values[15000::param_dataset.time_step,:],(param_dataset.dT,3),max_patches=param_dataset.NbTest)
        else:
            path_l63_dataset = param_dataset.path_l63_dataset#'../../Dataset4DVarNet/dataset_L63_with_noise.nc'
            genSuffixObs    = param_dataset.genSuffixObs#'JamesExp1'
                                
            ds_ncfile = xr.open_dataset(path_l63_dataset)
            dataTrainingNoNaN = ds_ncfile['x_train'].data
            dataTestNoNaN = ds_ncfile['x_test'].data
            
            if hasattr(ds_ncfile,'meanTr') == True :
                meanTr = ds_ncfile['meanTr']
                stdTr = ds_ncfile['stdTr']
    
                meanTr = float(meanTr.data)    
                stdTr = float(stdTr.data)
    
                dataTrainingNoNaN = stdTr * dataTrainingNoNaN + meanTr
                dataTestNoNaN     = stdTr * dataTestNoNaN + meanTr
            else:
                meanTr = 7.820 #np.mean( dataTrainingNoNaN )
                stdTr = 14.045 #np.std( dataTrainingNoNaN )            

            print(dataTrainingNoNaN.shape)
            print(dataTestNoNaN.shape)

            dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)
            dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
        
        if param_dataset.NbTraining < dataTrainingNoNaN.shape[0] :
            dataTrainingNoNaN = dataTrainingNoNaN[:param_dataset.NbTraining,:,:]
    
        if param_dataset.NbTest < dataTrainingNoNaN.shape[0] :
            dataTestNoNaN = dataTestNoNaN[:param_dataset.NbTest,:,:]
    
        # create missing data
        if param_dataset.flagTypeMissData == 0:
            print('..... Observation pattern: Random sampling of osberved L63 components')
            indRand         = np.random.permutation(dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2])
            indRand         = indRand[0:int(rateMissingData*len(indRand))]
            dataTraining    = np.copy(dataTrainingNoNaN).reshape((dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2],1))
            dataTraining[indRand] = float('nan')
            dataTraining    = np.reshape(dataTraining,(dataTrainingNoNaN.shape[0],dataTrainingNoNaN.shape[1],dataTrainingNoNaN.shape[2]))
            
            indRand         = np.random.permutation(dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2])
            indRand         = indRand[0:int(rateMissingData*len(indRand))]
            dataTest        = np.copy(dataTestNoNaN).reshape((dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2],1))
            dataTest[indRand] = float('nan')
            dataTest          = np.reshape(dataTest,(dataTestNoNaN.shape[0],dataTestNoNaN.shape[1],dataTestNoNaN.shape[2]))
        
            genSuffixObs    = genSuffixObs+'Rnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
        elif param_dataset.flagTypeMissData == 2:
            print('..... Observation pattern: Only the first L63 component is osberved')
            time_step_obs   = int(param_dataset.sampling_step)#int(1./(1.-rateMissingData))
            
            dataTraining    = np.zeros((dataTrainingNoNaN.shape))
            dataTraining[:] = float('nan')
            dataTraining[:,::time_step_obs,0] = dataTrainingNoNaN[:,::time_step_obs,0]
            
            dataTest    = np.zeros((dataTestNoNaN.shape))
            dataTest[:] = float('nan')
            dataTest[:,::time_step_obs,0] = dataTestNoNaN[:,::time_step_obs,0]
        
            genSuffixObs    = genSuffixObs+'Dim0_%02d_%02d'%(int(param_dataset.sampling_step),10*sigNoise**2)
           
        else:
            print('..... Observation pattern: All  L63 components osberved')
            time_step_obs   = int(param_dataset.sampling_step)#int(1./(1.-rateMissingData))
            dataTraining    = np.zeros((dataTrainingNoNaN.shape))
            dataTraining[:] = float('nan')
            dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
            
            dataTest    = np.zeros((dataTestNoNaN.shape))
            dataTest[:] = float('nan')
            dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]
        
            genSuffixObs    = genSuffixObs+'Sub_%02d_%02d'%(int(param_dataset.sampling_step),10*param_dataset.varNoise)
            
        # set to NaN patch boundaries
        dataTraining[:,0:10,:] =  float('nan')
        dataTest[:,0:10,:]     =  float('nan')
        dataTraining[:,param_dataset.dT-10:param_dataset.dT,:] =  float('nan')
        dataTest[:,param_dataset.dT-10:param_dataset.dT,:]     =  float('nan')
        
        # mask for NaN
        maskTraining = (dataTraining == dataTraining).astype('float')
        maskTest     = ( dataTest    ==  dataTest   ).astype('float')
        
        dataTraining = np.nan_to_num(dataTraining)
        dataTest     = np.nan_to_num(dataTest)
        
        # Permutation to have channel as #1 component
        dataTraining      = np.moveaxis(dataTraining,-1,1)
        maskTraining      = np.moveaxis(maskTraining,-1,1)
        dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)
        
        dataTest      = np.moveaxis(dataTest,-1,1)
        maskTest      = np.moveaxis(maskTest,-1,1)
        dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
        
        ############################################
        ## raw data
        X_train         = dataTrainingNoNaN
        X_train_missing = dataTraining
        mask_train      = maskTraining
        
        X_test         = dataTestNoNaN
        X_test_missing = dataTest
        mask_test      = maskTest
        
        ############################################
        ## normalized data
        #meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train) 
        #stdTr           = np.sqrt( np.mean( (X_train_missing-meanTr)**2 ) / np.mean(mask_train) )

        #if param_dataset.flagTypeMissData == 2:
        #    meanTr          = np.mean(X_train[:]) 
        #    stdTr           = np.sqrt( np.mean( (X_train-meanTr)**2 ) )
        
        #x_train_missing = ( X_train_missing - meanTr ) / stdTr
        #x_test_missing  = ( X_test_missing - meanTr ) / stdTr
        
        # scale wrt std
        
        x_train = (X_train - meanTr) / stdTr
        x_test  = (X_test - meanTr) / stdTr
        
        print('.... MeanTr = %.3f --- StdTr = %.3f '%(meanTr,stdTr))
        
        # Generate noisy observsation
        X_train_obs = X_train_missing + sigNoise * maskTraining * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[1],X_train_missing.shape[2])
        X_test_obs  = X_test_missing  + sigNoise * maskTest * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[1],X_test_missing.shape[2])
        
        x_train_obs = (X_train_obs - meanTr) / stdTr
        x_test_obs  = (X_test_obs - meanTr) / stdTr
        
        #print('..... Training dataset: %dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
        #print('..... Test dataset    : %dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
            
        # Initialization
        flagInit = 1
        mx_train = np.sum( np.sum( X_train , axis = 2 ) , axis = 0 ) / (X_train.shape[0]*X_train.shape[2])
        
        if flagInit == 0: 
          X_train_Init = mask_train * X_train_obs + (1. - mask_train) * (np.zeros(X_train_missing.shape) + meanTr)
          X_test_Init  = mask_test * X_test_obs + (1. - mask_test) * (np.zeros(X_test_missing.shape) + meanTr)
        else:
          X_train_Init = np.zeros(X_train.shape)
          for ii in range(0,X_train.shape[0]):
            # Initial linear interpolation for each component
            XInit = np.zeros((X_train.shape[1],X_train.shape[2]))
        
            for kk in range(0,3):
              indt  = np.where( mask_train[ii,kk,:] == 1.0 )[0]
              indt_ = np.where( mask_train[ii,kk,:] == 0.0 )[0]
        
              if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_train_obs[ii,kk,indt])
                XInit[kk,indt]  = X_train_obs[ii,kk,indt]
                XInit[kk,indt_] = fkk(indt_)
              else:
                XInit[kk,:] = XInit[kk,:] +  mx_train[kk]
        
            X_train_Init[ii,:,:] = XInit
        
          X_test_Init = np.zeros(X_test.shape)
          for ii in range(0,X_test.shape[0]):
            # Initial linear interpolation for each component
            XInit = np.zeros((X_test.shape[1],X_test.shape[2]))
        
            for kk in range(0,3):
              indt  = np.where( mask_test[ii,kk,:] == 1.0 )[0]
              indt_ = np.where( mask_test[ii,kk,:] == 0.0 )[0]
        
              if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_test_obs[ii,kk,indt])
                XInit[kk,indt]  = X_test_obs[ii,kk,indt]
                XInit[kk,indt_] = fkk(indt_)
              else:
                XInit[kk,:] = XInit[kk,:] +  mx_train[kk]
        
            X_test_Init[ii,:,:] = XInit
        
        
        x_train_Init = ( X_train_Init - meanTr ) / stdTr
        x_test_Init = ( X_test_Init - meanTr ) / stdTr
        
        # reshape to 2D tensors
        dT = param_dataset.dT
        x_train = x_train.reshape((-1,3,dT,1))
        mask_train = mask_train.reshape((-1,3,dT,1))
        x_train_Init = x_train_Init.reshape((-1,3,dT,1))
        x_train_obs = x_train_obs.reshape((-1,3,dT,1))
        
        x_test = x_test.reshape((-1,3,dT,1))
        mask_test = mask_test.reshape((-1,3,dT,1))
        x_test_Init = x_test_Init.reshape((-1,3,dT,1))
        x_test_obs = x_test_obs.reshape((-1,3,dT,1))
    
        print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    
    else:
        #flagTypeMissData = 2
        print('.... Load full dataset')
        
        path_l63_dataset = '../4dvarnet-forecast/dataset_L63_JamesExp1.nc'
        genSuffixObs    = 'JamesExp1'
                            
        ncfile = Dataset(path_l63_dataset,"r")
        x_train = ncfile.variables['x_train'][:]
        x_train_Init = ncfile.variables['x_train_Init'][:]
        x_train_obs = ncfile.variables['x_train_obs'][:]
        mask_train = ncfile.variables['mask_train'][:]
    
        x_test = ncfile.variables['x_test'][:]
        mask_test = ncfile.variables['mask_test'][:]
        x_test_Init = ncfile.variables['x_test_Init'][:]
        x_test_obs = ncfile.variables['x_test_obs'][:]
    
    
        meanTr = ncfile.variables['meanTr'][:]
        stdTr = ncfile.variables['stdTr'][:]
        meanTr = float(meanTr.data)    
        stdTr = float(stdTr.data)
            
        print('... meanTr/stdTr:')
        print(meanTr)
        print(stdTr)
        dT = param_dataset.dT
        
        x_train = x_train.reshape((-1,3,dT,1))
        mask_train = mask_train.reshape((-1,3,dT,1))
        x_train_Init = x_train_Init.reshape((-1,3,dT,1))
        x_train_obs = x_train_obs.reshape((-1,3,dT,1))
        
        x_test = x_test.reshape((-1,3,dT,1))
        mask_test = mask_test.reshape((-1,3,dT,1))
        x_test_Init = x_test_Init.reshape((-1,3,dT,1))
        x_test_obs = x_test_obs.reshape((-1,3,dT,1))
        
        X_train = stdTr * x_train.squeeze() + meanTr
        X_test = stdTr * x_test.squeeze() + meanTr
    
        print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
        print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

    data_train = X_train, x_train, mask_train, x_train_Init, x_train_obs
    data_test = X_test, x_test, mask_test, x_test_Init, x_test_obs
    stat_data = meanTr,stdTr

    return data_train,data_test,stat_data,genSuffixObs


def create_l63_forecast_datasets(param_dataset): 
    rateMissingData = (1-1./param_dataset.sampling_step)
    sigNoise = np.sqrt( param_dataset.varNoise )
    genSuffixObs = param_dataset.genSuffixObs
    
    ## Load or create L63 dataset
    if param_dataset.flag_generate_L63_data :
        ## data generation: L63 series

        class GD:
            model = 'Lorenz_63'
            class parameters:
                sigma = 10.0
                rho   = 28.0
                beta  = 8.0/3
            dt_integration = 0.01 # integration time
            dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
            #dt_obs = 8 # number of integration times between consecutive observations (for yo)
            #var_obs = np.array([0,1,2]) # indices of the observed variables
            nb_loop_train = 10**2 # size of the catalog
            nb_loop_test = 20000 # size of the true state and noisy observations
            #sigma2_catalog = 0.0 # variance of the model error to generate the catalog
            #sigma2_obs = 2.0 # variance of the observation error to generate observation

        GD = GD()    
        y0 = np.array([8.0,0.0,30.0])
        tt = np.arange(GD.dt_integration,GD.nb_loop_test*GD.dt_integration+0.000001,GD.dt_integration)
        #S = odeint(AnDA_Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[0.,5+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=np.arange(0,5+0.000001,GD.dt_integration),method='RK45')
        
        y0 = S.y[:,-1];
        S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[GD.dt_integration,GD.nb_loop_test+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=tt,method='RK45')
        S = S.y.transpose()
        
        
        ####################################################
        ## Generation of training and test dataset
        ## Extraction of time series of dT time steps            
          
        xt = time_series()
        xt.values = S
        xt.time   = tt
        # extract subsequences
        dataTrainingNoNaN = image.extract_patches_2d(xt.values[0:12000:param_dataset.time_step,:],(param_dataset.dT,3),max_patches=param_dataset.NbTraining)
        dataTestNoNaN     = image.extract_patches_2d(xt.values[15000::param_dataset.time_step,:],(param_dataset.dT,3),max_patches=param_dataset.NbTest)
    else:
        path_l63_dataset = param_dataset.path_l63_dataset#'../../Dataset4DVarNet/dataset_L63_with_noise.nc'
        genSuffixObs    = param_dataset.genSuffixObs#'JamesExp1'
                      
        
        ds_ncfile = xr.open_dataset(path_l63_dataset)
        dataTrainingNoNaN = ds_ncfile['x_train'].data
        dataTestNoNaN = ds_ncfile['x_test'].data
        
        meanTr = ds_ncfile['meanTr']
        stdTr = ds_ncfile['stdTr']

        meanTr = float(meanTr.data)    
        stdTr = float(stdTr.data)

        dataTrainingNoNaN = stdTr * dataTrainingNoNaN + meanTr
        dataTestNoNaN     = stdTr * dataTestNoNaN + meanTr
                  
        dataTrainingNoNaN = dataTrainingNoNaN[:,:,:param_dataset.dT]
        dataTestNoNaN = dataTestNoNaN[:,:,:param_dataset.dT]

        dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)
        dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
     
    if param_dataset.NbTraining < dataTrainingNoNaN.shape[0] :
        dataTrainingNoNaN = dataTrainingNoNaN[:param_dataset.NbTraining,:,:]

    if param_dataset.NbTest < dataTrainingNoNaN.shape[0] :
        dataTestNoNaN = dataTestNoNaN[:param_dataset.NbTest,:,:]

    # create missing data 
    if param_dataset.flagTypeMissData == 0:
        print('..... Observation pattern: Random sampling of osberved L63 components')
        
        if rateMissingData > 0. :
            indRand         = np.random.permutation(dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2])
            indRand         = indRand[0:int(rateMissingData*len(indRand))]
            dataTraining    = np.copy(dataTrainingNoNaN).reshape((dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2],1))
            dataTraining[indRand] = float('nan')
            dataTraining    = np.reshape(dataTraining,(dataTrainingNoNaN.shape[0],dataTrainingNoNaN.shape[1],dataTrainingNoNaN.shape[2]))
            
            indRand         = np.random.permutation(dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2])
            indRand         = indRand[0:int(rateMissingData*len(indRand))]
            dataTest        = np.copy(dataTestNoNaN).reshape((dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2],1))
            dataTest[indRand] = float('nan')
            dataTest          = np.reshape(dataTest,(dataTestNoNaN.shape[0],dataTestNoNaN.shape[1],dataTestNoNaN.shape[2]))
    
        else:
            dataTraining    = np.copy(dataTrainingNoNaN)
            dataTest        = np.copy(dataTestNoNaN)
    
        genSuffixObs    = genSuffixObs+'Rnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
    elif param_dataset.flagTypeMissData == 2:
        print('..... Observation pattern: Only the first L63 component is osberved')
        time_step_obs   = int(param_dataset.sampling_step)#int(1./(1.-rateMissingData))
        
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTraining[:,::time_step_obs,0] = dataTrainingNoNaN[:,::time_step_obs,0]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        dataTest[:,::time_step_obs,0] = dataTestNoNaN[:,::time_step_obs,0]
    
        genSuffixObs    = genSuffixObs+'Dim0_%02d_%02d'%(int(param_dataset.sampling_step),10*sigNoise**2)
       
    else:
        print('..... Observation pattern: All  L63 components osberved')
        time_step_obs   = int(param_dataset.sampling_step)#int(1./(1.-rateMissingData))
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]
    
        genSuffixObs    = genSuffixObs+'Sub_%02d_%02d'%(int(param_dataset.sampling_step),10*param_dataset.varNoise)
            
    # set to NaN the forecasting window
    idx_last_obs = param_dataset.dT - param_dataset.dt_forecast-1
    dataTraining[:,idx_last_obs+1:,:] =  float('nan')
    dataTest[:,idx_last_obs+1:,:]     =  float('nan')
    
    # mask for NaN
    maskTraining = (dataTraining == dataTraining).astype('float')
    maskTest     = ( dataTest    ==  dataTest   ).astype('float')
    
    dataTraining = np.nan_to_num(dataTraining)
    dataTest     = np.nan_to_num(dataTest)
    
    # Permutation to have channel as #1 component
    dataTraining      = np.moveaxis(dataTraining,-1,1)
    maskTraining      = np.moveaxis(maskTraining,-1,1)
    dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)
    
    dataTest      = np.moveaxis(dataTest,-1,1)
    maskTest      = np.moveaxis(maskTest,-1,1)
    dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
        
    ############################################
    ## raw data
    X_train         = dataTrainingNoNaN
    X_train_missing = dataTraining
    mask_train      = maskTraining
    
    X_test         = dataTestNoNaN
    X_test_missing = dataTest
    mask_test      = maskTest
        
    x_train = (X_train - meanTr) / stdTr
    x_test  = (X_test - meanTr) / stdTr
    
    print('.... MeanTr = %.3f --- StdTr = %.3f '%(meanTr,stdTr))
    
    # Generate noisy observsation
    X_train_obs = X_train_missing + sigNoise * maskTraining * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[1],X_train_missing.shape[2])
    X_test_obs  = X_test_missing  + sigNoise * maskTest * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[1],X_test_missing.shape[2])
    
    x_train_obs = (X_train_obs - meanTr) / stdTr
    x_test_obs  = (X_test_obs - meanTr) / stdTr
        
    # Initialization
    flagInit = 1
    mx_train = np.sum( np.sum( X_train , axis = 2 ) , axis = 0 ) / (X_train.shape[0]*X_train.shape[2])
    
    if flagInit == 0: 
      X_train_Init = mask_train * X_train_obs + (1. - mask_train) * (np.zeros(X_train_missing.shape) + meanTr)
      X_test_Init  = mask_test * X_test_obs + (1. - mask_test) * (np.zeros(X_test_missing.shape) + meanTr)
    else:
      X_train_Init = np.zeros(X_train.shape)
      for ii in range(0,X_train.shape[0]):
        # Initial linear interpolation for each component
        XInit = np.zeros((X_train.shape[1],X_train.shape[2]))
    
        for kk in range(0,3):
          indt  = np.where( mask_train[ii,kk,:] == 1.0 )[0]
          indt_ = np.where( mask_train[ii,kk,:] == 0.0 )[0]
    
          if len(indt) > 1:
            indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
            indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
            fkk = scipy.interpolate.interp1d(indt, X_train_obs[ii,kk,indt])
            XInit[kk,indt]  = X_train_obs[ii,kk,indt]
            XInit[kk,indt_] = fkk(indt_)
          else:
            XInit[kk,:] = XInit[kk,:] +  mx_train[kk]
    
        X_train_Init[ii,:,:] = XInit
    
      X_test_Init = np.zeros(X_test.shape)
      for ii in range(0,X_test.shape[0]):
        # Initial linear interpolation for each component
        XInit = np.zeros((X_test.shape[1],X_test.shape[2]))
    
        for kk in range(0,3):
          indt  = np.where( mask_test[ii,kk,:] == 1.0 )[0]
          indt_ = np.where( mask_test[ii,kk,:] == 0.0 )[0]
    
          if len(indt) > 1:
            indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
            indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
            fkk = scipy.interpolate.interp1d(indt, X_test_obs[ii,kk,indt])
            XInit[kk,indt]  = X_test_obs[ii,kk,indt]
            XInit[kk,indt_] = fkk(indt_)
          else:
            XInit[kk,:] = XInit[kk,:] +  mx_train[kk]
    
        X_test_Init[ii,:,:] = XInit
    
      print('........')
      print(X_train_Init[0:5,0,idx_last_obs]  )
      print(X_train[0:5,0,idx_last_obs]  )
    
      X_train_Init[:,:,idx_last_obs+1:] =  np.tile( X_train_Init[:,:,idx_last_obs].reshape((X_train_Init.shape[0],X_train_Init.shape[1],1)) , (1,1,param_dataset.dt_forecast) )
      X_test_Init[:,:,idx_last_obs+1:]  =  np.tile( X_test_Init[:,:,idx_last_obs].reshape((X_test_Init.shape[0],X_test_Init.shape[1],1)) , (1,1,param_dataset.dt_forecast) )
        
        
    x_train_Init = ( X_train_Init - meanTr ) / stdTr
    x_test_Init = ( X_test_Init - meanTr ) / stdTr
    
    
    print(x_train_Init[1,idx_last_obs:idx_last_obs+10,1])
    print(x_train_Init[100,idx_last_obs:idx_last_obs+10,1])
    
    # reshape to 2D tensors
    dT = param_dataset.dT
    x_train = x_train.reshape((-1,3,dT,1))
    mask_train = mask_train.reshape((-1,3,dT,1))
    x_train_Init = x_train_Init.reshape((-1,3,dT,1))
    x_train_obs = x_train_obs.reshape((-1,3,dT,1))
    
    x_test = x_test.reshape((-1,3,dT,1))
    mask_test = mask_test.reshape((-1,3,dT,1))
    x_test_Init = x_test_Init.reshape((-1,3,dT,1))
    x_test_obs = x_test_obs.reshape((-1,3,dT,1))

    print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    
    data_train = X_train, x_train, mask_train, x_train_Init, x_train_obs
    data_test = X_test, x_test, mask_test, x_test_Init, x_test_obs
    stat_data = meanTr,stdTr

    return data_train,data_test,stat_data,genSuffixObs


def create_l63_ode_solver_datasets(param_dataset): 
    sigNoise = np.sqrt( param_dataset.varNoise )
    genSuffixObs = param_dataset.genSuffixObs
    
    if param_dataset.dT < param_dataset.dT_test:
        dT = param_dataset.dT_test
    else:
        dT = param_dataset.dT
        
    ## Load or create L63 dataset
    if param_dataset.flag_generate_L63_data :
        ## data generation: L63 series

        class GD:
            model = 'Lorenz_63'
            class parameters:
                sigma = 10.0
                rho   = 28.0
                beta  = 8.0/3
            dt_integration = 0.01 # integration time
            dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
            #dt_obs = 8 # number of integration times between consecutive observations (for yo)
            #var_obs = np.array([0,1,2]) # indices of the observed variables
            nb_loop_train = 10**2 # size of the catalog
            nb_loop_test = 20000 # size of the true state and noisy observations
            #sigma2_catalog = 0.0 # variance of the model error to generate the catalog
            #sigma2_obs = 2.0 # variance of the observation error to generate observation

        GD = GD()    
        y0 = np.array([8.0,0.0,30.0])
        tt = np.arange(GD.dt_integration,GD.nb_loop_test*GD.dt_integration+0.000001,GD.dt_integration)
        #S = odeint(AnDA_Lorenz_63,x0,np.arange(0,5+0.000001,GD.dt_integration),args=(GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta));
        S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[0.,5+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=np.arange(0,5+0.000001,GD.dt_integration),method='RK45')
        
        y0 = S.y[:,-1];
        S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,GD.parameters.sigma,GD.parameters.rho,GD.parameters.beta),t_span=[GD.dt_integration,GD.nb_loop_test+0.000001],y0=y0,first_step=GD.dt_integration,t_eval=tt,method='RK45')
        S = S.y.transpose()
        
        
        ####################################################
        ## Generation of training and test dataset
        ## Extraction of time series of dT time steps            
          
        xt = time_series()
        xt.values = S
        xt.time   = tt
        # extract subsequences
        dataTrainingNoNaN = image.extract_patches_2d(xt.values[0:12000:param_dataset.time_step,:],(dT,3),max_patches=param_dataset.NbTraining)
        dataTestNoNaN     = image.extract_patches_2d(xt.values[15000::param_dataset.time_step,:],(dT,3),max_patches=param_dataset.NbTest)
    else:
        path_l63_dataset = param_dataset.path_l63_dataset#'../../Dataset4DVarNet/dataset_L63_with_noise.nc'
        genSuffixObs    = param_dataset.genSuffixObs#'JamesExp1'
                      
        
        ds_ncfile = xr.open_dataset(path_l63_dataset)
        dataTrainingNoNaN = ds_ncfile['x_train'].data
        dataTestNoNaN = ds_ncfile['x_test'].data
        
        meanTr = ds_ncfile['meanTr']
        stdTr = ds_ncfile['stdTr']

        meanTr = float(meanTr.data)    
        stdTr = float(stdTr.data)

        dataTrainingNoNaN = stdTr * dataTrainingNoNaN + meanTr
        dataTestNoNaN     = stdTr * dataTestNoNaN + meanTr
                  
        dataTrainingNoNaN = dataTrainingNoNaN[:,:,:param_dataset.time_step*dT:param_dataset.time_step]
        dataTestNoNaN = dataTestNoNaN[:,:,:param_dataset.time_step*dT:param_dataset.time_step]

        dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)
        dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
     
    if param_dataset.NbTraining < dataTrainingNoNaN.shape[0] :
        dataTrainingNoNaN = dataTrainingNoNaN[:param_dataset.NbTraining,:,:]

    if param_dataset.NbTest < dataTrainingNoNaN.shape[0] :
        dataTestNoNaN = dataTestNoNaN[:param_dataset.NbTest,:,:]

    # create missing data 
    print('..... Observation pattern: All  L63 components osberved')
    time_step_obs   = 1#int(1./(1.-rateMissingData))
    dataTraining    = np.zeros((dataTrainingNoNaN.shape))
    dataTraining[:] = float('nan')
    dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
    
    dataTraining    = np.copy( dataTrainingNoNaN )
    dataTest    = np.copy( dataTestNoNaN )

    genSuffixObs    = genSuffixObs+'odesolver_%02d'%(int(param_dataset.sampling_step))
            
    # set to NaN the forecasting window
    idx_last_obs = param_dataset.dT - param_dataset.dt_forecast-1
    dataTraining[:,idx_last_obs+1:,:] =  float('nan')
    dataTest[:,idx_last_obs+1:,:]     =  float('nan')
    
    # mask for NaN
    maskTraining = (dataTraining == dataTraining).astype('float')
    maskTest     = ( dataTest    ==  dataTest   ).astype('float')
    
    dataTraining = np.nan_to_num(dataTraining)
    dataTest     = np.nan_to_num(dataTest)
    
    # Permutation to have channel as #1 component
    dataTraining      = np.moveaxis(dataTraining,-1,1)
    maskTraining      = np.moveaxis(maskTraining,-1,1)
    dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)
    
    dataTest      = np.moveaxis(dataTest,-1,1)
    maskTest      = np.moveaxis(maskTest,-1,1)
    dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
        
    ############################################
    ## raw data
    X_train         = dataTrainingNoNaN
    X_train_missing = dataTraining
    mask_train      = maskTraining
    
    X_test         = dataTestNoNaN
    X_test_missing = dataTest
    mask_test      = maskTest
        
    x_train = (X_train - meanTr) / stdTr
    x_test  = (X_test - meanTr) / stdTr
    
    print('.... MeanTr = %.3f --- StdTr = %.3f '%(meanTr,stdTr))
    
    # Generate noisy observsation
    X_train_obs = X_train_missing 
    X_test_obs  = X_test_missing
    
    x_train_obs = (X_train_obs - meanTr) / stdTr
    x_test_obs  = (X_test_obs - meanTr) / stdTr
        
    # Initialization
    X_train_Init = 1. * X_train_missing
    X_test_Init = 1. * X_test_missing
                    
    x_train_Init = ( X_train_Init - meanTr ) / stdTr
    x_test_Init = ( X_test_Init - meanTr ) / stdTr
    
    # reshape to 2D tensors
    dT = param_dataset.dT
    x_train = x_train.reshape((-1,3,dT,1))
    mask_train = mask_train.reshape((-1,3,dT,1))
    x_train_Init = x_train_Init.reshape((-1,3,dT,1))
    x_train_obs = x_train_obs.reshape((-1,3,dT,1))
    
    x_test = x_test.reshape((-1,3,dT,1))
    mask_test = mask_test.reshape((-1,3,dT,1))
    x_test_Init = x_test_Init.reshape((-1,3,dT,1))
    x_test_obs = x_test_obs.reshape((-1,3,dT,1))

    print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
    print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))
    
    data_train = X_train, x_train, mask_train, x_train_Init, x_train_obs
    data_test = X_test, x_test, mask_test, x_test_Init, x_test_obs
    stat_data = meanTr,stdTr

    return data_train,data_test,stat_data,genSuffixObs



class BaseDataModule(pl.LightningDataModule):
    def __init__(self, input_data,param_datamodule):
        super().__init__()
        
        self.param_datamodule = param_datamodule

        self.input_data = input_data

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

        self.meanTr = None
        self.stdTr = None
        
        self.genSuffixObs = None
        self.dl_kw = param_datamodule.dl_kw
        self.setup()
        
        print( self.genSuffixObs )

    def setup(self, stage='test'):
        print()
        print('..... Setup datamodule',flush=True)
        data_train , data_test, stats_train, genSuffixObs = self.input_data #create_dataloaders(self.param_datamodule)#flag_load_data,flagTypeMissData,NbTraining,NbTest,time_step,dT,sigNoise,sampling_step)
        
        X_train, x_train, mask_train, x_train_Init, x_train_obs = data_train
        X_test, x_test, mask_test, x_test_Init, x_test_obs = data_test
        self.meanTr, self.stdTr = stats_train
        self.genSuffixObs = genSuffixObs
        
        # define dataloaders
        idx_val = x_train.shape[0]-500
            
        self.train_ds     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init[:idx_val:,:,:,:]),torch.Tensor(x_train_obs[:idx_val:,:,:,:]),torch.Tensor(mask_train[:idx_val:,:,:,:]),torch.Tensor(x_train[:idx_val:,:,:,:])) # create your datset
        self.val_ds         = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init[idx_val::,:,:,:]),torch.Tensor(x_train_obs[idx_val::,:,:,:]),torch.Tensor(mask_train[idx_val::,:,:,:]),torch.Tensor(x_train[idx_val::,:,:,:])) # create your datset
        self.test_ds         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Init),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset

    def norm_stats(self):
        return self.meanTr,self.stdTr

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
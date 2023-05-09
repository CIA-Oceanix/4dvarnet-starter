#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:32:46 2021

@author: rfablet
"""
import numpy as np
import matplotlib.pyplot as plt 
import os
#import tensorflow.keras as keras

import time
import copy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from netCDF4 import Dataset

from sklearn import decomposition

import src.solver_l63 as solver_4DVarNet
from src.data_l63 import BaseDataModule

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from sklearn.feature_extraction import image

import pytorch_lightning as pl
from omegaconf import OmegaConf

EPS_NORM_GRAD = 0. * 1.e-20  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_float32_matmul_precision('medium')
torch.set_float32_matmul_precision('high')

flag_load_data = False #  True#   

def get_constant_crop_l63(patch_dims, crop):
    
    print(patch_dims,flush=True)
    patch_weight = np.zeros(patch_dims, dtype="float32")
    mask = tuple(
        slice(crop[d], -crop[d]) if crop.get(d, 0) > 0 else slice(None, None)
        for d in range(0,3)
    )
    patch_weight[mask] = 1.0
    
    patch_weight = patch_weight / np.sum(patch_weight)
    return patch_weight


if 1*0:
    print('........ Data generation')
    flagRandomSeed = 0
    if flagRandomSeed == 0:
        print('........ Random seed set to 100')
        see_rnd = 200#100
        np.random.seed(see_rnd)
        torch.manual_seed(see_rnd)

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

def create_filename_ckpt(suffix,params_data,params_model):
    print(params_data)
    print(params_model)
    
    filename_chkpt = 'model-l63-' + params_data.genSuffixObs 
    filename_chkpt = filename_chkpt + '-Obs%02d'%params_data.sampling_step + '-Noise%02d'%(params_data.varNoise)        
    filename_chkpt = filename_chkpt + '-' + params_model.phi_param
    filename_chkpt = filename_chkpt + '-igrad%02d_%02d'%(params_model.n_grad,params_model.k_n_grad)+'-dgrad%d'%params_model.dim_grad_solver          
    #filename_chkpt = filename_chkpt + '-drop%02d'%(100*params_model.dropout)
    #filename_chkpt = filename_chkpt + '-rnd-init%02d'%(100*params_model.sig_rnd_init)
    #filename_chkpt = filename_chkpt + '-lstm-init%02d'%(100*params_model.sig_lstm_init)
    filename_chkpt = filename_chkpt + suffix
    
    
    print('.... filename: ' + filename_chkpt,flush=True)
    return filename_chkpt
    
def create_dataloaders(data_module): 
    rateMissingData = (1-1./data_module.sampling_step)
    sigNoise = np.sqrt( data_module.varNoise )
    genSuffixObs = data_module.genSuffixObs
    
    if data_module.flag_load_all_data == False :
     
        #data_module.flag_generate_L63_data = False    
        if data_module.flag_generate_L63_data :
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
            dataTrainingNoNaN = image.extract_patches_2d(xt.values[0:12000:data_module.time_step,:],(data_module.dT,3),max_patches=data_module.NbTraining)
            dataTestNoNaN     = image.extract_patches_2d(xt.values[15000::data_module.time_step,:],(data_module.dT,3),max_patches=data_module.NbTest)
        else:
            path_l63_dataset = data_module.path_l63_dataset#'../../Dataset4DVarNet/dataset_L63_with_noise.nc'
            genSuffixObs    = data_module.genSuffixObs#'JamesExp1'
                                
            ncfile = Dataset(path_l63_dataset,"r")
            dataTrainingNoNaN = ncfile.variables['x_train'][:]
            dataTestNoNaN = ncfile.variables['x_test'][:]
            
            meanTr = ncfile.variables['meanTr'][:]
            stdTr = ncfile.variables['stdTr'][:]
            meanTr = float(meanTr.data)    
            stdTr = float(stdTr.data)
        
            dataTrainingNoNaN = stdTr * dataTrainingNoNaN + meanTr
            dataTestNoNaN     = stdTr * dataTestNoNaN + meanTr
    
            dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)
            dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
    
    
        # create missing data
        if data_module.flagTypeMissData == 0:
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
        elif data_module.flagTypeMissData == 2:
            print('..... Observation pattern: Only the first L63 component is osberved')
            time_step_obs   = int(data_module.sampling_step)#int(1./(1.-rateMissingData))
            
            dataTraining    = np.zeros((dataTrainingNoNaN.shape))
            dataTraining[:] = float('nan')
            dataTraining[:,::time_step_obs,0] = dataTrainingNoNaN[:,::time_step_obs,0]
            
            dataTest    = np.zeros((dataTestNoNaN.shape))
            dataTest[:] = float('nan')
            dataTest[:,::time_step_obs,0] = dataTestNoNaN[:,::time_step_obs,0]
        
            genSuffixObs    = genSuffixObs+'Dim0_%02d_%02d'%(int(data_module.sampling_step),10*sigNoise**2)
           
        else:
            print('..... Observation pattern: All  L63 components osberved')
            time_step_obs   = int(data_module.sampling_step)#int(1./(1.-rateMissingData))
            dataTraining    = np.zeros((dataTrainingNoNaN.shape))
            dataTraining[:] = float('nan')
            dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
            
            dataTest    = np.zeros((dataTestNoNaN.shape))
            dataTest[:] = float('nan')
            dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]
        
            genSuffixObs    = genSuffixObs+'Sub_%02d_%02d'%(int(data_module.sampling_step),10*data_module.varNoise)
            
        # set to NaN patch boundaries
        dataTraining[:,0:10,:] =  float('nan')
        dataTest[:,0:10,:]     =  float('nan')
        dataTraining[:,data_module.dT-10:data_module.dT,:] =  float('nan')
        dataTest[:,data_module.dT-10:data_module.dT,:]     =  float('nan')
        
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
        meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train) 
        stdTr           = np.sqrt( np.mean( (X_train_missing-meanTr)**2 ) / np.mean(mask_train) )
        
        if data_module.flagTypeMissData == 2:
            meanTr          = np.mean(X_train[:]) 
            stdTr           = np.sqrt( np.mean( (X_train-meanTr)**2 ) )
        
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
        
        print('..... Training dataset: %dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
        print('..... Test dataset    : %dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))
            
        import scipy
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
        dT = data_module.dT
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
        dT = data_module.dT
        
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

#print('........ Define AE architecture')
#shapeData  = x_train.shape[1:]
# freeze all ode parameters

class Phi_ode(torch.nn.Module):
    def __init__(self,meanTr,stdTr):
          super(Phi_ode, self).__init__()
          self.sigma = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
          self.rho    = torch.nn.Parameter(torch.Tensor([np.random.randn()]))
          self.beta   = torch.nn.Parameter(torch.Tensor([np.random.randn()]))

          self.sigma  = torch.nn.Parameter(torch.Tensor([10.]))
          self.rho    = torch.nn.Parameter(torch.Tensor([28.]))
          self.beta   = torch.nn.Parameter(torch.Tensor([8./3.]))

          self.dt        = 0.01
          self.IntScheme = 'rk4'
          self.stdTr     = stdTr
          self.meanTr    = meanTr                      
    def _odeL63(self, xin):
        x1  = xin[:,0,:]
        x2  = xin[:,1,:]
        x3  = xin[:,2,:]
        
        dx_1 = (self.sigma*(x2-x1)).view(-1,1,xin.size(2))
        dx_2 = (x1*(self.rho-x3)-x2).view(-1,1,xin.size(2))
        dx_3 = (x1*x2 - self.beta*x3).view(-1,1,xin.size(2))
        
        dpred = torch.cat((dx_1,dx_2,dx_3),dim=1)
        return dpred

    def _EulerSolver(self, x):
        return x + self.dt * self._odeL63(x)

    def _RK4Solver(self, x):
        k1 = self._odeL63(x)
        x2 = x + 0.5 * self.dt * k1
        k2 = self._odeL63(x2)
      
        x3 = x + 0.5 * self.dt * k2
        k3 = self._odeL63(x3)
          
        x4 = x + self.dt * k3
        k4 = self._odeL63(x4)

        return x + self.dt * (k1+2.*k2+2.*k3+k4)/6.
  
    def forward(self, x):
        X = self.stdTr * x.view(-1,x.size(1),x.size(2))
        X = X + self.meanTr
        
        if self.IntScheme == 'euler':
            xpred = self._EulerSolver( X[:,:,0:x.size(2)-1] )
        else:
            xpred = self._RK4Solver( X[:,:,0:x.size(2)-1] )

        xpred = xpred - self.meanTr
        xpred = xpred / self.stdTr

        xnew  = torch.cat((x[:,:,0].view(-1,x.size(1),1),xpred),dim=2)
        
        xnew = xnew.view(-1,x.size(1),x.size(2),1)
        
        return xnew

class Phi_unet(torch.nn.Module):
    def __init__(self,shapeData,DimAE,dW=5):
        super(Phi_unet, self).__init__()
        self.pool1  = torch.nn.AvgPool2d((4,1))
        #self.conv1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
        self.conv1  = torch.nn.Conv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
        self.conv2  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        
        self.conv21 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        self.conv22 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        self.conv23 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        self.conv3  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        #self.conv4 = torch.nn.Conv1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,1,padding=0,bias=False)

        self.conv2Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0],(4,1),stride=(4,1),bias=False)          
        #self.conv5 = torch.nn.Conv1d(2*shapeData[0]*DimAE,2*shapeData[0]*DimAE,3,padding=1,bias=False)
        #self.conv6 = torch.nn.Conv1d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)
        #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)

        #self.convHR1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
        #self.convHR1  = ConstrainedConv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
        self.convHR1  = torch.nn.Conv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
        self.convHR2  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        
        self.convHR21 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        self.convHR22 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        self.convHR23 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
        self.convHR3  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)

        self.shapeData = shapeData
    def forward(self, xinp):
        #x = self.fc1( torch.nn.Flatten(x) )
        #x = self.pool1( xinp )
        x = self.pool1( xinp )
        x = self.conv1( x )
        x = self.conv2( F.relu(x) )
        x = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
        x = self.conv3( x )
        x = self.conv2Tr( x )
        #x = self.conv5( F.relu(x) )
        #x = self.conv6( F.relu(x) )
        
        xHR = self.convHR1( xinp )
        xHR = self.convHR2( F.relu(xHR) )
        xHR = torch.cat((self.convHR21(xHR), self.convHR22(xHR) * self.convHR23(xHR)),dim=1)
        xHR = self.convHR3( xHR )
        
        x   = x + xHR #torch.add(x,1.,xHR)
        
        x = x.view(-1,self.shapeData[0],self.shapeData[1],1)
        return x


class Model_H(torch.nn.Module):
    def __init__(self,shapeData):
        super(Model_H, self).__init__()
        #self.DimObs = 1
        #self.dimObsChannel = np.array([shapeData[0]])
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 1
        self.dimObsChannel = np.array([shapeData[0]])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout

class HParam:
    def __init__(self,phi='unet',
                 n_grad = 10,
                 k_n_grad = 1,
                 dim_grad_solver = 10,
                 w_loss=1.,
                 automatic_optimization=True,
                 ):
        self.lr       = []
        self.n_grad          = 1
        self.dim_grad_solver = 10
        self.dropout         = 0.25
        self.w_loss          = []
        self.automatic_optimization = True
        self.hparams.shapeData = [3,200,1]

        self.alpha_proj    = 0.5
        self.alpha_mse = 10.

        self.k_batch = 1
        #self.hparams.n_grad          = 5#self.hparams.n_grad#5#self.hparams.nb_grad_update[0]
        #self.hparams.k_n_grad        = 1#self.hparams.k_n_grad#1
        #self.hparams.dim_grad_solver = dimGradSolver
        #self.hparams.dropout         = rateDropout



class Lit4dVarNet_L63(pl.LightningModule):
    def __init__(self,params=None,patch_weight=None,stats_training_data=None,*args, **kwargs):
        super().__init__()
        #self.hparams = HParam() if params is None else params
        #hparam = {} if params is None else params
        #hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)
        #hparams = hparam

        #print(hparams,flush=True)
        self.save_hyperparameters(params)
        #self.save_hyperparameters({**hparams, **kwargs})

        print(self.hparams,flush=True)
        
        #self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(patch_weight), requires_grad=False) if patch_weight is not None else 1.
        self.hparams.automatic_optimization = True# False#

        #m_NormObs = solver_4DVarNet.Model_WeightedL2Norm()        
        #m_NormPhi = solver_4DVarNet.Model_WeightedL2Norm()


        # main model
        if self.hparams.phi_param == 'ode':
            self.model        = solver_4DVarNet.GradSolver_with_rnd(Phi_ode(), 
                                                                    Model_H(self.hparams.shapeData), 
                                                                    solver_4DVarNet.model_Grad(self.hparams.shapeData, self.hparams.UsePeriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode='zeros'), 
                                                                    None, None, 
                                                                    #solver_4DVarNet.Model_Var_Cost2(m_NormObs, m_NormPhi, self.hparams.ShapeData,1,np.array([self.hparams.shapeData[0]])),
                                                                    self.hparams.shapeData, self.hparams.n_grad, EPS_NORM_GRAD,self.hparams.lr_grad,self.hparams.lr_rnd)#, self.hparams.eps_norm_grad)
        elif self.hparams.phi_param == 'unet':
            self.model        = solver_4DVarNet.GradSolver_with_rnd(Phi_unet(self.hparams.shapeData,self.hparams.DimAE), 
                                                                    Model_H(self.hparams.shapeData), 
                                                                    solver_4DVarNet.model_Grad(self.hparams.shapeData, self.hparams.UsePeriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout, padding_mode='zeros'), 
                                                                    None, None, 
                                                                    #solver_4DVarNet.Model_Var_Cost2(m_NormObs, m_NormPhi, self.hparams.ShapeData,1,np.array([self.hparams.shapeData[0]])),
                                                                    self.hparams.shapeData, self.hparams.n_grad, EPS_NORM_GRAD,self.hparams.lr_grad,self.hparams.lr_rnd)#, self.hparams.eps_norm_grad)
        
        self.w_loss  = 1.#torch.nn.Parameter(torch.Tensor(patch_weight), requires_grad=False) if patch_weight is not None else 1.
        self.x_rec   = None # variable to store output of test method
        self.x_rec_obs = None
        self.x_gt   = None # variable to store output of test method
        #self.curr = 0

        self.set_norm_stats = stats_training_data if stats_training_data is not None else (0.0,1.)
        self._set_norm_stats()
        
        self.automatic_optimization = self.hparams.automatic_optimization
        
    def forward(self):
        return 1

    def configure_optimizers(self):
        #optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
        #                              {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
        #                            {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.hparams.lr_update[0]},
        #                            ], lr=0.)
        optimizer    = torch.optim.Adam(self.model.parameters(), self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-5, last_epoch=- 1, verbose=False)

        return [optimizer],[lr_scheduler]
    
    def _set_norm_stats(self):
        self.meanTr = self.set_norm_stats[0]
        self.stdTr = self.set_norm_stats[1]        
        #print(' mean/std: %f -- %f'%(self.meanTr,self.stdTr))
        
    def on_train_epoch_start(self):
        self.model.n_grad = self.hparams.n_grad 
        self.model.n_step = self.hparams.k_n_grad * self.model.n_grad
        
        self.model.model_Grad.sig_lstm_init = self.hparams.sig_lstm_init
        self.model.lr_rnd = self.hparams.lr_rnd
        self.model.lr_grad = self.hparams.lr_grad
        
        self._set_norm_stats()
    
    def on_test_epoch_start(self):
        #torch.inference_mode(mode=False)
        self.x_rec = None
        self.x_gt  = None
        self.x_obs = None

        self.model.n_grad   = self.hparams.n_grad 
        self.model.n_step = self.hparams.k_n_grad * self.model.n_grad

        self.model.model_Grad.sig_lstm_init = self.hparams.sig_lstm_init
        self.model.lr_rnd = self.hparams.lr_rnd
        self.model.lr_grad = self.hparams.lr_grad

        self._set_norm_stats()
        
        print('--- n_grad = %d -- k_n_grad = %d -- n_step = %d'%(self.model.n_grad,self.hparams.k_n_grad,self.model.n_step) )
        print('--- ')
    def on_validation_epoch_start(self):
        self.x_rec = None

        self.model.n_grad = self.hparams.n_grad 
        self.model.n_step = self.hparams.k_n_grad * self.model.n_grad

        self.model.model_Grad.sig_lstm_init = self.hparams.sig_lstm_init
        self.model.lr_rnd = self.hparams.lr_rnd
        self.model.lr_grad = self.hparams.lr_grad

        self._set_norm_stats()
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
                    
        # compute loss and metrics
        loss, out, metrics = self.compute_loss(train_batch, phase='train')
        
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(train_batch, phase='train',batch_init=out[0],hidden=out[1],cell=out[2],normgrad=out[3],prev_iter=(kk+1)*self.model.n_grad)
            loss = loss + loss1
        
        self.log("tr_mse", self.stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # initial grad value
        if self.hparams.automatic_optimization == False :
            # backward
            self.manual_backward(loss)
        
            if (batch_idx + 1) % self.hparams.k_batch == 0:
                # optimisation step
                opt.step()
                
                # grad initialization to zero
                opt.zero_grad()
         
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(val_batch, phase='val',batch_init=out[0],hidden=out[1],cell=out[2],normgrad=out[3],prev_iter=(kk+1)*self.model.n_grad)
            loss = loss1

        self.log('val_loss', self.stdTr**2 * metrics['mse'] )
        self.log("val_mse", self.stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        _,inputs_obs,___,targets_GT = test_batch
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
    
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(test_batch, phase='test',batch_init=out[0].detach(),hidden=out[1],cell=out[2],normgrad=out[3],prev_iter=(kk+1)*self.model.n_grad)

        self.log("test_mse", self.stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.x_rec is None :
            self.x_rec = out[0].squeeze(dim=-1).detach().cpu().numpy() * self.stdTr + self.meanTr
            self.x_gt  = targets_GT.squeeze(dim=-1).detach().cpu().numpy() * self.stdTr + self.meanTr
            self.x_obs = inputs_obs.squeeze(dim=-1).detach().cpu().numpy() * self.stdTr + self.meanTr
        else:
            self.x_rec = np.concatenate((self.x_rec,out[0].squeeze(dim=-1).detach().cpu().numpy() * self.stdTr + self.meanTr),axis=0)
            self.x_gt  = np.concatenate((self.x_gt,targets_GT.squeeze(dim=-1).detach().cpu().numpy() * self.stdTr + self.meanTr),axis=0)
            self.x_obs  = np.concatenate((self.x_obs,inputs_obs.squeeze(dim=-1).detach().cpu().numpy() * self.stdTr + self.meanTr),axis=0)
                
    
    def on_test_epoch_end(self):
        mse = np.mean( (self.x_rec - self.x_gt)**2 )

        print('... mse test: %.3f '%mse)

    def compute_loss(self, batch, phase, batch_init = None , hidden = None , cell = None , normgrad = 0.0,prev_iter=0):
        #print('... Inference mode')
        #print( torch.is_inference_mode_enabled() )
        #print('-----------------')
 
        with torch.set_grad_enabled(True):
            inputs_init_,inputs_obs,masks,targets_GT = batch
     
            #inputs_init = inputs_init_
            if batch_init is None :
                inputs_init = inputs_init_
            else:
                inputs_init = batch_init + self.hparams.sig_rnd_init *  torch.randn( batch_init.size() ).to(device)
                
            if phase == 'train' :                
                inputs_init = inputs_init.detach()
            
            # with torch.set_grad_enabled(phase == 'train'):
            #inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            #outputs, hidden_new, cell_new, normgrad_ = self.model(inputs_init_, inputs_obs, masks)#,hidden = hidden , cell = cell , normgrad = normgrad)
            #outputs, hidden_new, cell_new, normgrad_ = self.model(inputs_init, inputs_obs, masks ,hidden = None , cell = None , normgrad = normgrad )            
            outputs, hidden_new, cell_new, normgrad_ = self.model(inputs_init, inputs_obs, masks, hidden = hidden , cell = cell , normgrad = normgrad, prev_iter = prev_iter )

            #loss_mse   = solver_4DVarNet.compute_WeightedLoss((outputs - targets_GT), self.w_loss)
            loss_mse = torch.mean((outputs - targets_GT) ** 2)
            loss_prior = torch.mean((self.model.phi_r(outputs) - outputs) ** 2)
            loss_prior_gt = torch.mean((self.model.phi_r(targets_GT) - targets_GT) ** 2)

            loss = self.hparams.alpha_mse * loss_mse
            loss += 0.5 * self.hparams.alpha_prior * (loss_prior + loss_prior_gt)
            
            # metrics
            mse       = loss_mse.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            if (phase == 'val') or (phase == 'test'):                
                outputs = outputs.detach()#.requires_grad_(True)
        
        out = [outputs,hidden_new, cell_new, normgrad_]
        
        return loss,out, metrics

class HParam_FixedPoint:
    def __init__(self):
        self.n_iter_fp       = 1
        self.k_n_fp       = 1

        self.alpha_proj    = 0.5
        self.alpha_mse = 1.
        self.lr = 1.e-3

class LitModel_FixedPoint(pl.LightningModule):
    def __init__(self,conf=HParam_FixedPoint(),*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # hyperparameters
        self.hparams.n_iter_fp    = 5
        self.hparams.k_n_fp        = 1
                
        self.hparams.alpha_prior    = 0.5
        self.hparams.alpha_mse = 1.e1        
        self.hparams.lr    = 1.e-3

        # main model
        self.model        = Phi_unet()
        self.x_rec    = None # variable to store output of test method
        self.x_rec_obs = None
        self.curr = 0
        

    def forward(self):
        return 1

    def configure_optimizers(self):
        optimizer   = optim.Adam([{'params': self.model.parameters(), 'lr': self.hparams.lr},
                                    ], lr=0.)
        return optimizer
    
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        # compute loss and metrics
        loss, out, metrics = self.compute_loss(train_batch, phase='train')
        
        for kk in range(0,self.hparams.k_n_fp-1):
            loss1, out, metrics = self.compute_loss(train_batch, phase='train',batch_init=out)
            loss = loss + loss1
        
        self.log("tr_mse", self.stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                 
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')
        for kk in range(0,self.hparams.k_n_fp-1):
            loss1, out, metrics = self.compute_loss(val_batch, phase='val',batch_init=out)
            loss = loss1

        self.log('val_loss', self.stdTr**2 * metrics['mse'] )
        self.log("val_mse", self.stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
        
        for kk in range(0,self.hparams.k_n_fp-1):
            loss1, out, metrics = self.compute_loss(test_batch, phase='test',batch_init=out)

        self.log("test_mse", self.stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'preds': out.detach().cpu()}

    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        x_test_rec = self.stdTr * x_test_rec + self.meanTr        
        self.x_rec = x_test_rec.squeeze()

        return [{'mse':0.,'preds': 0.}]

    def compute_loss(self, batch, phase, batch_init = None ):

        inputs_init_,inputs_obs,masks,targets_GT = batch
        
        print(inputs_init_,flush=True)
        print(targets_GT,flush=True)
 
        if batch_init is None :
            inputs_init = inputs_init_
        else:
            inputs_init = batch_init
            
        if phase == 'train' :                
            inputs_init = inputs_init.detach()
            
        with torch.set_grad_enabled(True):
            outputs = inputs_init
            
            for kk in range(0,self.hparams.n_iter_fp-1):
               outputs = inputs_obs * masks + (1. - masks) * self.model( outputs ) 
            outputs = self.model( outputs ) 

            loss_mse = torch.mean((outputs - targets_GT) ** 2)
            loss_prior = torch.mean((self.model(outputs) - outputs) ** 2)
            loss_prior_gt = torch.mean((self.model(targets_GT) - targets_GT) ** 2)

            loss = self.hparams.alpha_mse * loss_mse
            loss += 0.5 * self.hparams.alpha_prior * (loss_prior + loss_prior_gt)
            
            # metrics
            mse       = loss_mse.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            if (phase == 'val') or (phase == 'test'):                
                outputs = outputs.detach()
                
        return loss,outputs, metrics
    
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from pathlib import Path

def get_cfg(xp_cfg, overrides=None):
    overrides = overrides if overrides is not None else []
    def get():
        cfg = hydra.compose(config_name='main', overrides=
            [
                f'xp={xp_cfg}',
                'file_paths=jz',
                'entrypoint=train',
            ] + overrides
        )

        return cfg
    try:
        with hydra.initialize_config_dir(str(Path('hydra_config').absolute())):
            return get()
    except ValueError as e:
        return get()

if __name__ == '__main__':
        
    #cfg = get_cfg("base")
    # cfg = get_cfg("xp_aug/xp_repro/quentin_repro")
    cfg = OmegaConf.load('config/xp/base_l63.yaml')
    print(OmegaConf.to_yaml(cfg))

    dm = BaseDataModule(cfg.datamodule.param_datamodule)
    
    mod = Lit4dVarNet_L63(cfg.model.params,patch_weight=get_constant_crop_l63(patch_dims=cfg.model.params.w_loss.patch_dims,crop=cfg.model.params.w_loss.crop))
    mod.load_from_checkpoint('outputs/2023-05-07/22-59-30/base_l63/checkpoints/val_mse=0.6534-epoch=379.ckpt')

    mod.set_norm_stats = dm.norm_stats()

    mod.meanTr = dm.meanTr
    mod.stdTr  = dm.stdTr
    
    print('n_step = %d'%mod.model.n_step)
    profiler_kwargs = {'max_epochs': 400 }

    suffix_exp = 'exp%02d'%cfg.datamodule.param_datamodule.flagTypeMissData+cfg.params.suffix_exp
    
    
    filename_chkpt = 'model-l63-'+ dm.genSuffixObs        
    filename_chkpt = filename_chkpt+cfg.params.phi_param+'-'              
    filename_chkpt = filename_chkpt + suffix_exp+'-Noise%02d'%(cfg.datamodule.param_datamodule.varNoise)


    filename_chkpt = filename_chkpt+'-igrad%02d_%02d'%(mod.hparams.n_grad,mod.hparams.k_n_grad)+'-dgrad%d'%cfg.params.dim_grad_solver
    filename_chkpt = filename_chkpt+'-drop%02d'%(100*cfg.params.dropout)
    filename_chkpt = filename_chkpt+'-rnd-init%02d'%(100*mod.hparams.sig_rnd_init)
    filename_chkpt = filename_chkpt+'-lstm-init%02d'%(100*mod.hparams.sig_lstm_init)

    print('.... chkpt: '+filename_chkpt)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath= './resL63/'+suffix_exp,
                                          filename= filename_chkpt + '-{epoch:02d}-{val_loss:.2f}',
                                          save_top_k=3,
                                          mode='min')
    trainer = pl.Trainer(devices=1,accelerator="gpu",  **profiler_kwargs,callbacks=[checkpoint_callback])
    trainer.fit(mod, datamodule=dm ) #dataloaders['train'], dataloaders['val'])        
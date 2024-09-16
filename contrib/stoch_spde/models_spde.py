import einops
import xarray as xr
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats
from contrib.stoch_spde.spde import *
from contrib.stoch_spde.scipy_sparse_tools import *
from contrib.stoch_spde.unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Upsampler(torch.nn.Module):
    def __init__(self, scale_factor, mode, align_corners, antialias, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor,
                          mode=self.mode, align_corners=self.align_corners,
                          antialias=self.antialias)
        return x

def sparse_eye(size, val = torch.tensor(1.0)):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size).to(device)
    if len(val.size())==0:
        values = (val.expand(size)).to(device)
    else:
        values = val.to(device)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size])).to(device)

def sparse_repeat(size,n1,n2):
    """
    Returns a sparse zero-filled tensor repeating 
    a 2D size*size sparse zero-filled tensor
    n1 times along dim1 and n2 times along n2
    """
    return torch.sparse.LongTensor(size*n1,size*n2).to(device)

    
class Encoder(torch.nn.Module):

    def __init__(self,shape_data):
        super(Encoder, self).__init__()
        self.n_t, self.n_y, self.n_x  = shape_data
        self.unet = UNet(enc_chs=(self.n_t,8,16),
                          pools=(2,2),
                          dec_chs=(16,8),
                          pads=(0,0),
                          kernel_size=3)
        self.cnn = torch.nn.Conv2d(self.n_t,8*self.n_t,(3,3),padding=1,bias=False)

    def forward(self, x):
        # input shape (b,t,y,x) --> output shape (b,7*t,y,x)
        #theta = self.unet(x)
        theta = self.cnn(x)
        # activation functions on some parameters
        return theta

class Prior_SPDE(torch.nn.Module):

    def __init__(self, shape_data, st_lag, trainable_lag=False, pow=1, spde_type="adv_diff", scheme="FDM", colored_noise=False):

        super(Prior_SPDE, self).__init__()
        self.n_t, self.n_y, self.n_x = shape_data

        if not trainable_lag:
            self.dx, self.dy, self.dt = torch.tensor(st_lag[1]), torch.tensor(st_lag[2]), torch.tensor(st_lag[0])
        else:
            self.dt = torch.tensor(st_lag[0]) #torch.abs(torch.nn.Parameter(torch.tensor(10.), requires_grad=True))
            self.dy = torch.tensor(st_lag[1]) #torch.abs(torch.nn.Parameter(torch.tensor(10.), requires_grad=True))
            self.dx = torch.tensor(st_lag[2]) #torch.abs(torch.nn.Parameter(torch.tensor(10.), requires_grad=True))
       
        self.nb_nodes = self.n_x*self.n_y
        self.Id = sparse_eye(self.nb_nodes)
        self.Id2 = sparse_eye(self.nb_nodes*self.n_t)
        self.pow = pow
        self.alpha  = int(2*self.pow)
        self.nu = int(self.alpha -1)
        self.spde_type = spde_type
        self.scheme = scheme
        self.colored_noise = colored_noise
        
    def custom_sigmoid(self,min,max):
        add_const = min/(max-min)
        mult_const = max-min
        self.dx = torch.nn.Parameter(F.relu(self.dx.data)+0.01, requires_grad=True)
        self.dy = torch.nn.Parameter(F.relu(self.dy.data)+0.01, requires_grad=True)

    def stack_indices(self,mat,ir,ic,row,col,val):

        row_ = mat.coalesce().indices()[0]
        col_ = mat.coalesce().indices()[1]
        val_ = mat.coalesce().values()
        row = torch.cat((row,row_+ir))
        col = torch.cat((col,col_+ic))
        val = torch.cat((val,val_))
        return row, col, val

    def create_Q(self,
                 kappa,
                 m,
                 H,
                 tau,
                 store_block_diag=False,
                 return_noise=False):

        n_b = H.shape[0]

        # set regularization variance term to appropriate size
        if torch.is_tensor(tau):
            #torch.full(tau.size(),1.)
            tau = torch.squeeze(tau,dim=1)
        else:
            tau = torch.stack(n_b*\
                              [torch.stack(self.nb_nodes*\
                                          [torch.stack(self.n_t*[torch.tensor(tau)])]\
                                          )]\
                              ).to(device)
        
        # initialize outputs    
        Q = list()
        Qs = list()
        if store_block_diag==True:
            block_diag=list()
            
        if self.scheme=="FDM":
            self.fdm = DiffOperator_FDM  
        if self.scheme=="FUDM1":
            self.fdm = DiffOperator_FUDM1   
        if self.scheme=="FUDM3":
            self.fdm = DiffOperator_FUDM3     
        
        for batch in range(n_b):
  
            # initialize Qs (noise precision matrix)
            Qsb = list()
            for k in range(self.n_t):
                if self.colored_noise==False:
                    Qs_ = sparse_eye(self.nb_nodes) 
                else:
                    if self.spde_type!="adv":
                        Qs_ = sparse_eye(self.nb_nodes)
                    else:
                        Qs_ = self.fdm(self.n_x,self.n_y,self.dx,self.dy,
                                  None,
                                  H[batch,:,:,:,k],
                                  kappa[batch,:,:,k])
                        Qs_ = pow_diff_operator(Qs_.to(device),2,split=False,sparse=True)
                        Qs_ = spspmm(Qs_.t(),Qs_)
                Qsb.append(Qs_)  
            
            # Build model evolution and noise effects operator
            inv_M = list() # linear model evolution matrix (inverse)
            inv_S = list() # T*Tt with T the noise effect matrix (inverse) 
            Q_list = list()
            for k in range(self.n_t):
                if self.spde_type=="adv_diff":
                    A = self.fdm(self.n_x,self.n_y,self.dx,self.dy,
                                         m[batch,:,:,k],
                                         H[batch,:,:,:,k],
                                         kappa[batch,:,:,k])
                elif self.spde_type=="diff":
                    A = self.fdm(self.n_x,self.n_y,self.dx,self.dy,
                                     None,
                                     H[batch],
                                     kappa)
                elif self.spde_type=="adv":
                    A = self.fdm(self.n_x,self.n_y,self.dx,self.dy,
                                     m[batch,:,:,k],
                                     torch.zeros(H[batch,:,:,:,k].shape).to(H.device),#None,
                                     torch.zeros(kappa[batch,:,:,k].shape).to(kappa.device))#kappa)
                else:
                    A = DiffOperator_Isotropic(self.n_x,self.n_y,self.dx,self.dy,
                                     kappa)
                if self.alpha>1:
                    B = pow_diff_operator(A.to(device),self.alpha,split=False,sparse=True)
                else:
                    B = A
                #B = A

                # initialize Q0 = P0^-1 = cov(x0)
                if k==0:
                    inv_tau_0 = sparse_eye(self.nb_nodes,(1./tau[batch,:,0])*torch.sqrt(self.dt))
                    Qs_tilde0 = spspmm(spspmm(inv_tau_0.t(),
                                              Qsb[0]),
                                       inv_tau_0)
                    Q0 = (self.dx*self.dy)*spspmm(spspmm(B.t(),
                                                         Qs_tilde0),
                                                  B)
                    inv_M0 = self.Id+self.dt*B
                    inv_S0 = spspmm(spspmm(inv_M0.t(),
                                           Qs_tilde0),
                                    inv_M0)
                    #Q0 = (1./2)*(Q0+Q0.t()) + 5e-2*sparse_eye(self.nb_nodes)
                else:
                    inv_tau_k = sparse_eye(self.nb_nodes,(1./tau[batch,:,k])*torch.sqrt(self.dt))
                    Qs_tilde = spspmm(spspmm(inv_tau_k.t(),
                                             Qsb[k]),
                                      inv_tau_k)
                    Q_k = (self.dx*self.dy)*spspmm(spspmm(B.t(),
                                                         Qs_tilde),
                                                  B)
                    inverse_M = self.Id+self.dt*B
                    inverse_S = spspmm(spspmm(inverse_M.t(),
                                              Qs_tilde),#Qs),
                                       inverse_M)
                    inv_M.append(inverse_M)
                    inv_S.append(inverse_S)
                    Q_list.append(Q_k)
 
            if store_block_diag==True:
                l = list(inv_S)
                l.insert(0,inv_S0)

                block_diag.append(l)
                
                    
            # Build the global precision matrix
            row = torch.tensor([]).to(device)
            col = torch.tensor([]).to(device) 
            val = torch.tensor([]).to(device)    
             
            # first line 
            inv_tau = sparse_eye(self.nb_nodes,(1./tau[batch,:,1])*torch.sqrt(self.dt))
            Qs_tilde = spspmm(spspmm(inv_tau.t(),Qsb[0]),inv_tau)
            row, col, val = self.stack_indices(inv_S0+Qs_tilde,
                                          0,0, 
                                          row,col,val) 
            row, col, val = self.stack_indices(-1.*spspmm(Qs_tilde,inv_M[0]), 
                                          0,self.nb_nodes, 
                                          row,col,val) 
            # loop 
            for i in np.arange(1,self.n_t-1): 
                inv_tau_1 = sparse_eye(self.nb_nodes,(1./tau[batch,:,i])*torch.sqrt(self.dt)) 
                Qs_tilde1 = spspmm(spspmm(inv_tau_1.t(),Qsb[i]),inv_tau_1) 
                inv_tau_2 = sparse_eye(self.nb_nodes,(1./tau[batch,:,i+1])*torch.sqrt(self.dt))
                Qs_tilde2 = spspmm(spspmm(inv_tau_2.t(),Qsb[i+1]),inv_tau_2) 
                row, col, val = self.stack_indices(-1.*spspmm(inv_M[i-1].t(),Qs_tilde1), 
                                              i*self.nb_nodes,(i-1)*self.nb_nodes, 
                                              row,col,val) 
                row, col, val = self.stack_indices(spspmm(spspmm(inv_M[i-1].t(),Qs_tilde1),inv_M[i-1])+Qs_tilde2, 
                                              i*self.nb_nodes,i*self.nb_nodes, 
                                              row,col,val) 
                row, col, val = self.stack_indices(-1.*spspmm(Qs_tilde2,inv_M[i]), 
                                              i*self.nb_nodes,(i+1)*self.nb_nodes, 
                                              row,col,val) 
            # last line 
            inv_tau = sparse_eye(self.nb_nodes,(1./tau[batch,:,self.n_t-1])*torch.sqrt(self.dt))
            Qs_tilde = spspmm(spspmm(inv_tau.t(),Qsb[self.n_t-1]),inv_tau) 
            row, col, val = self.stack_indices(-1.*spspmm(inv_M[self.n_t-2].t(),Qs_tilde), 
                                          (self.n_t-1)*self.nb_nodes,(self.n_t-2)*self.nb_nodes, 
                                          row,col,val) 
            row, col, val = self.stack_indices(spspmm(spspmm(inv_M[self.n_t-2].t(),Qs_tilde),inv_M[self.n_t-2]), 
                                          (self.n_t-1)*self.nb_nodes,(self.n_t-1)*self.nb_nodes,row,col,val) 
             
            # create sparse tensor 
            index = torch.stack([row, col], dim=0) 
            value = val 
            Qg = torch.sparse_coo_tensor(index.long(), value,  
                                           torch.Size([self.n_t*self.nb_nodes, 
                                                       self.n_t*self.nb_nodes]),
                                        requires_grad=True).coalesce().to(device) 

            #Qg = pow_diff_operator(Qg,pow=self.alpha,sparse=True,split=True,n_t=self.n_t)
            
            # add batch
            Q.append(Qg)
            Qs.append(Qsb)

        if store_block_diag==True:
            # Q has size #batch*(nt*nbnodes)*(nt*nbnodes)
            # block_diag is list of size #batch
            if return_noise:
                return Q, block_diag, Qs
            else:
                return Q, block_diag
        else:
            if return_noise:
                return Q, Qs
            else:
                return Q

    def forward(self,
                kappa,
                m,
                H,
                tau,
                store_block_diag=False,
                return_noise=False):

        res = self.create_Q(kappa, m, H, tau,
                          store_block_diag=store_block_diag,
                          return_noise=return_noise)
        return res

class NLL(torch.nn.Module):

    def __init__(self, shape_data, st_lag=[1,1,1], pow=1, spde_type="adv_diff", scheme="FUDM1",crop=None, downsamp=None):

        super(NLL,self).__init__()
        self.spde_type = spde_type
        self.scheme = scheme
        self.pow = pow        
        self.crop = crop
        self.downsamp = downsamp
        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )
        
        if crop is not None:
            shape_data = [shape_data[0],
                          shape_data[1]-self.crop,
                          shape_data[2]-self.crop]
            
        if downsamp is not None:
            shape_data = [shape_data[0],
                          shape_data[1]//self.downsamp,
                          shape_data[2]//self.downsamp] 
        
        self.encoder = Encoder(shape_data)
        
        self.operator_spde = Prior_SPDE(shape_data,st_lag,pow=self.pow,
                                        spde_type=self.spde_type,
                                        scheme=self.scheme)
        
    def custom_sigmoid(self,x,min,max):
        add_const = min/(max-min)
        mult_const = max-min
        return (torch.sigmoid(x)+add_const)*mult_const
        
    def reshape_params(self,theta):

        n_b, n_t, n_x, n_y = theta.shape
        n_t = theta.shape[1]//8

        # reshape the parameters
        kappa = theta[:,:n_t,:,:]
        tau = theta[:,n_t:2*n_t,:,:]
        m1 = theta[:,2*n_t:3*n_t,:,:]
        m2 = theta[:,3*n_t:4*n_t,:,:]
        vx = theta[:,4*n_t:5*n_t,:,:]
        vy = theta[:,5*n_t:6*n_t,:,:]
        gamma = theta[:,6*n_t:7*n_t,:,:]
        beta = theta[:,7*n_t:8*n_t,:,:]
        
        H = []
        for k in range(n_t):
            vx_ = torch.reshape(vx[:,k,:,:],(n_b,n_x*n_y))
            vy_ = torch.reshape(vy[:,k,:,:],(n_b,n_x*n_y))
            vxy = torch.stack([vx_,vy_],dim=2)
            vxyT = torch.permute(vxy,(0,2,1))
            gamma_ = torch.reshape(gamma[:,k,:,:],(n_b,n_x*n_y))
            beta_ = torch.reshape(beta[:,k,:,:],(n_b,n_x*n_y))
            H_ = torch.einsum('ij,bk->bijk',
                              torch.eye(2).to(device),
                              gamma_)+\
                 torch.einsum('bk,bijk->bijk',beta_,torch.einsum('bki,bjk->bijk',vxy,vxyT))
            H.append(H_)
        H = torch.stack(H,dim=4)
        m = torch.stack([m1,m2],dim=1)

        kappa = torch.permute(kappa,(0,2,3,1))
        tau = torch.permute(tau,(0,2,3,1))
        m = torch.permute(m,(0,1,3,4,2))
        H = torch.reshape(H,(n_b,2,2,n_x,n_y,n_t))
        H = torch.permute(H,(0,1,2,3,4,5))

        # reshaping n_x,n_y -> n_x*n_y
        kappa = torch.reshape(kappa,(n_b,1,n_y*n_x,n_t))
        m = torch.reshape(m,(n_b,2,n_y*n_x,n_t))
        H = torch.reshape(H,(n_b,2,2,n_y*n_x,n_t))
        tau = torch.reshape(tau,(n_b,1,n_y*n_x,n_t))

        #H = torch.full(H.shape,0.).to(device)
        return [kappa, m, H, tau]
    
    def downsamp_params(self, kappa, m, H, tau, sp_dims):
        
        n_b, _, nb_nodes, n_t = m.shape
        n_x, n_y = sp_dims
        
        kappa = torch.reshape(kappa,(n_b,1,n_y,n_x,n_t))
        tau = torch.reshape(tau,(n_b,1,n_y,n_x,n_t))
        m = torch.reshape(m,(n_b,2,n_y,n_x,n_t))
        H = torch.reshape(H,(n_b,2,2,n_y,n_x,n_t))
        
        kappa = torch.unsqueeze(torch.permute(
                                 self.down(torch.permute(kappa[:,0,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1)
        tau = torch.unsqueeze(torch.permute(
                                 self.down(torch.permute(tau[:,0,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1)    
        m1 = torch.unsqueeze(torch.permute(
                                 self.down(torch.permute(m[:,0,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        m2 = torch.unsqueeze(torch.permute(
                                 self.down(torch.permute(m[:,1,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        m = torch.cat([m1,m2],dim=1)
        h11 = torch.unsqueeze(torch.permute(
                                 self.down(torch.permute(H[:,0,0,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        h12 = torch.unsqueeze(torch.permute(
                                 self.down(torch.permute(H[:,0,1,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        h22 = torch.unsqueeze(torch.permute(
                                 self.down(torch.permute(H[:,1,1,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        H = torch.reshape(torch.cat([h11,h12,h12,h22],dim=1),(n_b,2,2,n_y//self.downsamp,n_x//self.downsamp,n_t))
        
        kappa = torch.reshape(kappa,(n_b,1,nb_nodes//(self.downsamp**2),n_t))
        tau = torch.reshape(tau,(n_b,1,nb_nodes//(self.downsamp**2),n_t))
        m = torch.reshape(m,(n_b,2,nb_nodes//(self.downsamp**2),n_t))
        H = torch.reshape(H,(n_b,2,2,nb_nodes//(self.downsamp**2),n_t))
        
        return kappa, m, H, tau
    
    def upsamp_params(self, kappa, m, H, tau, sp_dims):
        
        n_b, _, nb_nodes, n_t = m.shape
        n_x, n_y = sp_dims
        
        kappa = torch.reshape(kappa,(n_b,1,n_y,n_x,n_t))
        tau = torch.reshape(tau,(n_b,1,n_y,n_x,n_t))
        m = torch.reshape(m,(n_b,2,n_y,n_x,n_t))
        H = torch.reshape(H,(n_b,2,2,n_y,n_x,n_t))
        
        kappa = torch.unsqueeze(torch.permute(
                                 self.up(torch.permute(kappa[:,0,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1)
        tau = torch.unsqueeze(torch.permute(
                                 self.up(torch.permute(tau[:,0,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1)    
        m1 = torch.unsqueeze(torch.permute(
                                 self.up(torch.permute(m[:,0,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        m2 = torch.unsqueeze(torch.permute(
                                 self.up(torch.permute(m[:,1,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        m = torch.cat([m1,m2],dim=1)
        h11 = torch.unsqueeze(torch.permute(
                                 self.up(torch.permute(H[:,0,0,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        h12 = torch.unsqueeze(torch.permute(
                                 self.up(torch.permute(H[:,0,1,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        h22 = torch.unsqueeze(torch.permute(
                                 self.up(torch.permute(H[:,1,1,:,:,:],(0,3,1,2))),(0,2,3,1)),dim=1) 
        H = torch.reshape(torch.cat([h11,h12,h12,h22],dim=1),(n_b,2,2,int(n_y*self.downsamp),
                                                                      int(n_x*self.downsamp),n_t))

        kappa = torch.reshape(kappa,(n_b,1,int(nb_nodes*(self.downsamp**2)),n_t))
        tau = torch.reshape(tau,(n_b,1,int(nb_nodes*(self.downsamp**2)),n_t))
        m = torch.reshape(m,(n_b,2,int(nb_nodes*(self.downsamp**2)),n_t))
        H = torch.reshape(H,(n_b,2,2,int(nb_nodes*(self.downsamp**2)),n_t))
        
        return kappa, m, H, tau
    

    def forward(self, x, theta=None, mu=None, det=True):

        n_b, n_t, n_x, n_y = x.shape

        # compute the parameters
        if theta is None:
            theta = self.encoder(x)
            
        if not isinstance(theta, list):
            kappa, m, H, tau = self.reshape_params(theta)
        else:
            kappa, m, H, tau = theta
        
        # compute Q(theta)
        if self.crop is not None:
            c = self.crop//2
            x = x[:,:,c:-c,c:-c]
            if mu is not None:
                mu = mu[:,:,c:-c,c:-c]
            kappa = torch.reshape(torch.reshape(kappa,(n_b,1,n_y,n_x,n_t))[:,:,c:-c,c:-c,:],
                                   (n_b,1,(n_x-self.crop)*(n_y-self.crop),n_t))
            tau = torch.reshape(torch.reshape(tau,(n_b,1,n_y,n_x,n_t))[:,:,c:-c,c:-c,:],
                                   (n_b,1,(n_x-self.crop)*(n_y-self.crop),n_t))
            m = torch.reshape(torch.reshape(m,(n_b,2,n_y,n_x,n_t))[:,:,c:-c,c:-c,:],
                                   (n_b,2,(n_x-self.crop)*(n_y-self.crop),n_t))
            H = torch.reshape(torch.reshape(H,(n_b,2,2,n_y,n_x,n_t))[:,:,:,c:-c,c:-c,:],
                                   (n_b,2,2,(n_x-self.crop)*(n_y-self.crop),n_t))
            n_b, n_t, n_y, n_x = x.shape
            
        if self.downsamp is not None:
            kappa, m, H, tau = self.downsamp_params(kappa, m, H, tau, sp_dims=[n_y, n_x])
            
        Q, block_diag = self.operator_spde(kappa,
                                           m,
                                           H,
                                           tau,
                                           store_block_diag=True)
        
        # compute determinant(Q)
        if det==True:
            det_Q = list()
            for i in range(n_b):
                log_det = 0.
                for j in range(0,len(block_diag[i])):
                    BD = block_diag[i][j].to_dense().to(device)#.type(torch.LongTensor)
                    chol_block, info  = torch.linalg.cholesky_ex(BD)
                    if info!=0:
                        return torch.tensor([np.nan for _ in range(n_b)])
                    log_det_block =  2*torch.sum(\
                                         torch.log(\
                                         torch.diagonal(\
                                         chol_block,
                                         0)\
                                        )\
                                       )
                    log_det += log_det_block
                # log(det(Q^k)) = sum_k=1^p log(det(Q))
                #for _ in range(2,(2*self.pow)+1):
                #    log_det += log_det
                det_Q.append(log_det)

        # compute Mahanalobis distance xT.Q.x
        MD = list()
        for i in range(n_b):
            if self.downsamp is None:
                if mu is None:
                    MD_ = sp_mm(Q[i],torch.reshape(x[i],(n_t*n_x*n_y,1)))
                    MD_ = torch.matmul(torch.reshape(x[i],(1,n_t*n_x*n_y)),MD_)
                else:
                    MD_ = sp_mm(Q[i],torch.reshape(x[i]-mu[i],(n_t*n_x*n_y,1)))
                    MD_ = torch.matmul(torch.reshape(x[i]-mu[i],(1,n_t*n_x*n_y)),MD_)
            else:
                if mu is None:
                    MD_ = sp_mm(Q[i],torch.reshape(self.down(x)[i],(n_t*n_x*n_y//(self.downsamp**2),1)))
                    MD_ = torch.matmul(torch.reshape(self.down(x)[i],(1,n_t*n_x*n_y//(self.downsamp**2))),MD_)
                else:
                    MD_ = sp_mm(Q[i],torch.reshape(self.down(x)[i]-self.down(mu)[i],(n_t*n_x*n_y//(self.downsamp**2),1)))
                    MD_ = torch.matmul(torch.reshape(self.down(x)[i]-self.down(mu)[i],(1,n_t*n_x*n_y//(self.downsamp**2))),MD_)
            MD.append(MD_[0,0])

        # Negative log-likelihood
        if det==True:
            log_det = torch.stack(det_Q)
        MD = torch.stack(MD)

        # NLL is -log[p_theta(x)] has size (#batch,1)
        # GradNLL (with autodiff) will have same dim as x
        if det==True:
            NLL = -1.*(log_det - MD)
        else:
            NLL = -1.*MD
            
        if self.downsamp is not None:
            kappa, m, H, tau = self.upsamp_params(kappa, m, H, tau, sp_dims=[n_y//self.downsamp, n_x//self.downsamp])
            

        return NLL

class NLpObs(torch.nn.Module):

    def __init__(self, noise=1e-3):
        super(NLpObs, self).__init__()
        self.noise = noise

    def forward(self, x, batch, transpose=False):

        n_b, n_t, n_x, n_y = x.shape
        mask = batch.input.isfinite()
        if transpose:
            mask = torch.permute(mask,(0,1,3,2))
            inp = torch.permute(batch.input,(0,1,3,2))
            x = torch.permute(x,(0,1,3,2))
        else:
            inp = batch.input
        nb_nodes = n_x*n_y

        NLobs = list()
        for i in range(n_b):
            obs = inp[i][mask[i]]
            # define sparse observation operator
            row = []
            col = []
            idx = 0
            for k in range(n_t):
                idD = torch.where(torch.flatten(mask[i,k,:,:])!=0.)[0]
                if len(idD)>0:
                    row.extend( (idx + np.arange(0,len(idD))).tolist() )
                    col.extend( ((k*nb_nodes)+idD).tolist() )
                    idx = idx + len(idD)
            val = np.ones(len(row))
            nb_obs = len(torch.where(torch.flatten(mask[i])!=0.)[0])
            opH = torch.sparse_coo_tensor(torch.LongTensor([row,col]),
                                         torch.FloatTensor(val),
                                         torch.Size([nb_obs,n_t*nb_nodes])).coalesce().to(device)
            
            inv_R = (1./self.noise)*sparse_eye(nb_obs)
            m = torch.unsqueeze(obs,dim=1)-sp_mm(opH,torch.unsqueeze(torch.flatten(x[i]),dim=1))
            NLobs.append(torch.matmul(torch.transpose(m,0,1),sp_mm(inv_R,m))[0,0])
            
        NLobs = -1.*torch.stack(NLobs)

        return NLobs

# NN formulations of the prior in 4DVarNet

class BilinAEPriorCost(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):
        super().__init__()
        self.nt = nt
        self.bilin_quad = bilin_quad
        self.conv_in = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.bilin_1 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_21 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = torch.nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def forward_ae(self, x):
        x = self.down(x)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state, exclude_params=False):
        if not exclude_params:
            return F.mse_loss(state, self.forward_ae(state))
        else:
            return F.mse_loss(state[:,:self.nt,:,:], self.forward_ae(state)[:,:self.nt,:,:])

# Additional NN Prior in replacement of BilinAE
class PriorNet(torch.nn.Module):
    def __init__(self, dim_in,  dim_hidden, nparam, kernel_size=3, downsamp=None):

        super().__init__()

        self.nt = dim_in
        self.nparam = nparam
        self.dim_param = self.nt * nparam
        #self.dim_hidden_param = dim_hidden*self.dim_param
        self.dim_hidden_param = 100
        self.bilin_quad = False

        # state estimation
        self.conv_in = torch.nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_1 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_21 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = torch.nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_out = torch.nn.Conv2d(
            2 * dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # parameter estimation
        self.conv_in_param = torch.nn.Conv2d(
            self.dim_param, self.dim_hidden_param, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden_param = torch.nn.Conv2d(
            self.dim_hidden_param, self.dim_hidden_param, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.add_grad_info = torch.nn.Conv2d(
            self.dim_hidden_param + 3*self.nt, self.dim_hidden_param, kernel_size=kernel_size, padding=kernel_size // 2
        )

        # rebuild the augmented state
        self.aug_conv_out = torch.nn.Conv2d(
            dim_in + self.dim_hidden_param, dim_in + self.dim_param, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = torch.nn.AvgPool2d(downsamp) if downsamp is not None else torch.nn.Identity()
        self.up = (
            torch.nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else torch.nn.Identity()
        )

    def forward(self, state):
        
        state = self.down(state)
        n_b, _, n_y, n_x = state.shape
        

        # state estimation
        x = self.conv_in(state[:,:self.nt,:,:])
        x = self.conv_hidden(F.relu(x))
        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )

        # parameter estimation
        theta = self.conv_in_param(state[:,self.nt:,:,:])
        theta = self.conv_hidden_param(F.relu(theta))
        theta = self.add_grad_info(torch.cat([x, 
                                              torch.reshape(kfilts.spatial_gradient(x,normalized=True),
                                                            (n_b, 2*self.nt, n_y, n_x)),
                                              theta], dim=1))

        # final state
        state = self.aug_conv_out(
            torch.cat([x, theta], dim=1)
        )
        state = self.up(state)
        return state

class AugPriorCost(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, nparam, kernel_size=3, downsamp=None):

        super().__init__()
        self.nt = dim_in
        self.priornet = PriorNet(dim_in, dim_hidden, nparam, kernel_size, downsamp)

    def forward_ae(self, x):
        x = self.priornet(x)
        return x

    def forward(self, state, exclude_params=False):
        if not exclude_params:
            return F.mse_loss(state, self.forward_ae(state))
        else:
            return F.mse_loss(state[:,:self.nt,:,:], self.forward_ae(state)[:,:self.nt,:,:])


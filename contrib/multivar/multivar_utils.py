import numpy as np
from src.utils import get_constant_crop
import torch


def get_multivar_aug_dims(multivar_dict):
    aug_dims = None
    dim = 0

    for var, var_info in multivar_dict.items():
        if 'aug' in var_info.keys():
            if var_info.aug:
                if aug_dims is None:
                    aug_dims = []
    # THE AUGMENTATION WILL ALWAYS WORK ON THE NEXT DIMENSION ONLY
                aug_dims.append((dim, dim+1))

        if 'mask_path' in var_info.keys():
            dim += 1
        dim+=1

    return aug_dims

def get_multivar_prior_dims_in(multivar_dict, channels_per_dim):

    dims_in = 0
    for var, var_info in multivar_dict.items():
        if var_info.input_arch == 'prior_input' or var_info.output_arch == 'full_output':
            dims_in+=1
    return dims_in * channels_per_dim

def get_multivar_prior_dims_out(multivar_dict, channels_per_dim):

    dims_in = 0
    for var, var_info in multivar_dict.items():
        if var_info.output_arch == 'full_output':
            dims_in+=1
    return dims_in * channels_per_dim

def get_multivar_grad_dims(multivar_dict, channels_per_dim):

    dims_in = 0
    for var, var_info in multivar_dict.items():
        if var_info.output_arch == 'full_output':
            dims_in+=1
    return dims_in * channels_per_dim

def get_multivar_triang_time_wei(patch_dims, dims_out, offset=0, **crop_kw):
    patch_dims = patch_dims.copy()
    patch_dims["time"] = dims_out
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )

def get_multivar_forecast_wei(patch_dims, dims_out, **crop_kw):
    """
    return weight for forecast reconstruction:
    patch_dims: dimension of the patches used

    linear from 0 to 1 where there are obs
    linear from 1 to 0.5 for 7 days of forecast
    0 elsewhere
    """
    pw = get_constant_crop(patch_dims, **crop_kw)
    time_patch_weight = np.concatenate(
        (np.linspace(0, 1, (patch_dims['time'] - 1) // 2),
         np.linspace(1, 0.5, 7),
         np.zeros((patch_dims['time'] + 1) // 2 - 7)),
        axis=0)
    
    # assuming dims_out = time * n_vars_out
    n_vars_out = dims_out // patch_dims['time']
    time_patch_weight = np.concatenate([time_patch_weight]*n_vars_out, axis=0)
    pw = np.concatenate([pw]*n_vars_out, axis=0)

    final_patch_weight = time_patch_weight[:, None, None] * pw
    return final_patch_weight

def get_multivar_mapping_wei(patch_dims, dims_out, offset=0, **crop_kw):
    """
    return weight for forecast reconstruction:
    patch_dims: dimension of the patches used

    linear from 0 to 1 where there are obs
    linear from 1 to 0.5 for 7 days of forecast
    0 elsewhere
    """
    pw = get_constant_crop(patch_dims, **crop_kw)
    time_patch_weight = np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )
    
    # assuming dims_out = time * n_vars_out
    n_vars_out = dims_out // patch_dims['time']
    time_patch_weight = np.concatenate([time_patch_weight]*n_vars_out, axis=0)
    pw = np.concatenate([pw]*n_vars_out, axis=0)

    final_patch_weight = time_patch_weight[:, None, None] * pw
    return final_patch_weight

class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class MultivarBatchSelector(metaclass=SingletonMeta):

    def __init__(self):
        super().__init__()

    def mask_batch(self, batch):
        masked_batch = batch
        masked_dim_size = masked_batch.shape[2]
        masked_batch[:,self.full_input_idx, masked_dim_size//2:] = np.nan
        masked_batch[:,self.prior_input_idx, masked_dim_size//2:] = np.nan

        return masked_batch

    def multivar_setup(self, multivar_info):

        self.full_input_idx = torch.Tensor(multivar_info['full_input_idx']).type(torch.int64).cuda()
        self.prior_input_idx = torch.Tensor(multivar_info['prior_input_idx']).type(torch.int64).cuda()
        self.full_output_idx = torch.Tensor(multivar_info['full_output_idx']).type(torch.int64).cuda()
        self.state_obs_channels = torch.Tensor(multivar_info['state_obs_channels']).type(torch.int64).cuda()
        self.state_obs_input_idx = torch.Tensor(multivar_info['state_obs_input_idx']).type(torch.int64).cuda()
        self.var_names = multivar_info['var_names']
        self.output_var_names = [self.var_names[idx] for idx in self.full_output_idx.tolist()]

        print('full_input_idx: {}'.format(list(self.full_input_idx)))
        print('prior_input_idx: {}'.format(list(self.prior_input_idx)))
        print('full_output_idx: {}'.format(list(self.full_output_idx)))
        print('state_obs_channels: {}'.format(list(self.state_obs_channels)))
        print('state_obs_input_idx: {}'.format(list(self.state_obs_input_idx)))

    def multivar_full_input(self, batch):
        new_batch = torch.index_select(batch, dim=1, index=self.full_input_idx)
        new_batch = new_batch.view(new_batch.shape[0], -1, *new_batch.shape[-2:])
        #print('multivar_full_input batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)

    def multivar_prior_input(self, batch):
        new_batch = torch.index_select(batch, dim=1, index=self.prior_input_idx)
        new_batch = new_batch.view(new_batch.shape[0], -1, *new_batch.shape[-2:])
        #print('multivar_prior_input batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)
    
    def multivar_full_output(self, batch):
        new_batch = torch.index_select(batch, dim=1, index=self.full_output_idx)
        new_batch = new_batch.view(new_batch.shape[0], -1, *new_batch.shape[-2:])
        #print('new batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)
    
    def multivar_state_obs(self, state):
        new_batch = torch.index_select(state, dim=1, index=self.state_obs_channels)
        #print('multivar_state_obs batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)
    
    def multivar_obs_input(self, batch):
        new_batch = torch.index_select(batch, dim=1, index=self.state_obs_input_idx)
        new_batch = new_batch.view(new_batch.shape[0], -1, *new_batch.shape[-2:])
        #print('multivar_obs_input batch shape: {} | type: {}'.format(new_batch.shape, new_batch.dtype))
        return new_batch.type(torch.float)
    
    def multivar_output_var_names(self):
        return self.output_var_names
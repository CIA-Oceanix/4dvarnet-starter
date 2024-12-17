from contrib.moving_patches.movpatch_data_tests import MovingPatchDataModuleFastRecGPU, XrDatasetMovingPatchFastRecGPU
from contrib.multivar.multivar_utils import MultivarBatchSelector
import numpy as np
import xarray as xr
import functools as ft

class MultivarXrDataset(XrDatasetMovingPatchFastRecGPU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    #def __getitem__(self, item):
    #    item = super().__getitem__(item)
    #    print('item shape: {}'.format(item.shape))
    #    return item

class MultivarDataModule(MovingPatchDataModuleFastRecGPU):

    def __init__(self, multivar_da, domains, xrds_kw, dl_kw, norm_stats=None, **kwargs):
        self.input_da, self.multivar_information = multivar_da
        super().__init__(self.input_da, domains, xrds_kw, dl_kw, norm_stats=norm_stats, **kwargs)
        self.multivar_info()

    def norm_stats(self):
        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
            print("Norm stats", self._norm_stats)
        return self._norm_stats
    
    def output_norm_stats(self):
        return self._norm_stats[0][self.multivar_info_dict['full_output_idx']], self._norm_stats[1][self.multivar_info_dict['full_output_idx']]
    
    def input_norm_stats(self):
        return self._norm_stats[0][self.multivar_info_dict['full_input_idx']], self._norm_stats[1][self.multivar_info_dict['full_input_idx']]

    def train_mean_std(self):
        m = []
        s = []
        data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])

        for var, var_information in self.multivar_information.items():
            if var.startswith('masked_'):
                m_var, s_var = data.sel(variable=var.split('masked_')[1]).pipe(lambda da: (da.mean().values.item(), da.std().values.item()))
            else:
                m_var, s_var = data.sel(variable=var).pipe(lambda da: (da.mean().values.item(), da.std().values.item()))
            m.append(m_var)
            s.append(s_var)
        return np.array(m), np.array(s)

    def post_fn(self):
        
        m, s = self.norm_stats()
        def normalize(item):
            item_shape_len = len(item.shape)
            return (item - np.expand_dims(m, tuple(range(1, item_shape_len)))) / np.expand_dims(s, tuple(range(1, item_shape_len)))

        return normalize

    def setup(self, stage='test'):
        # calling MovingPatch Datasets, rand=True for train only
        post_fn = self.post_fn()
        self.train_ds = MultivarXrDataset(
            self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn, rand=True
        )
        self.val_ds = MultivarXrDataset(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )
        self.test_ds = MultivarXrDataset(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn, rand=False
        )

    def multivar_info(self):
        full_input_idx = []
        prior_input_idx = []
        full_output_idx = []
        state_obs_channels = []
        state_obs_input_idx = []
        len_time_channels = self.xrds_kw.patch_dims['time']

        for idx, (var, var_information) in enumerate(self.multivar_information.items()):
            if var_information['input_arch'] == 'full_input':
                full_input_idx.append(idx)
                if var_information['output_arch'] == 'full_output':
                    state_obs_input_idx.append(idx)
                if 'masked_obs' in list(var_information.keys()):
                    state_obs_input_idx.append(idx+1)
            elif var_information['input_arch'] == 'prior_input':
                prior_input_idx.append(idx)
            if var_information['output_arch'] == 'full_output':
                full_output_idx.append(idx)

        for state_obs_input in state_obs_input_idx:
            if state_obs_input in list(full_output_idx):
                idx = np.argwhere(np.array(full_output_idx) == state_obs_input).item()
                state_obs_channels.extend([idx*len_time_channels+i for i in range(len_time_channels)])
                
        multivar_info = dict(
            full_input_idx = full_input_idx,
            prior_input_idx = prior_input_idx,
            full_output_idx = full_output_idx,
            state_obs_channels = state_obs_channels,
            state_obs_input_idx = state_obs_input_idx,
        )
        self.multivar_info_dict = multivar_info

        batch_selector = MultivarBatchSelector()
        batch_selector.multivar_setup(multivar_info)


            
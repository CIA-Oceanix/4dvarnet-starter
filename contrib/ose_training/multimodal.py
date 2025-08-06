import collections
import torch


TrainingItem = collections.namedtuple( 'TrainingItem', ['input', 'input_coords', 'sst', 'tgt', 'tgt_coords'])

class MultiModalOseDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, sst_path='../sla-data-registry/mur_pp.nc'):
        self.ds = ds
        self.sst = xr.open_dataset(sst_path)



    



import torch.nn as nn
import torch.nn.functional as F
import einops

class Wrapper(nn.Module):
    def __init__(self, mod, sst=False):
        super().__init__()
        self.mod = mod
        self.sst = sst

    def forward(self, batch):
        if not self.sst:
            inp = batch.input.nan_to_num()
            inp = einops.rearrange(inp, 'b t y x -> b t () y x')
        else:
            inp = torch.stack([
                batch.input.nan_to_num(),
                batch.sst.nan_to_num(),
            ], dim=2)
        out = self.mod(inp)
        out = einops.rearrange(out, 'b t () y x -> b t y x')
        return out

    def prior_cost(self, *args, **kw):
        return 0.

    def init_state(self, *args, **kw):
        return 0.

class PriorWrapper(nn.Module):
    def __init__(self, mod, sst=False):
        super().__init__()
        self.mod = mod

    def forward_ae(self, x):
        inp = einops.rearrange(x, 'b t y x -> b t () y x')
        out = self.mod(inp)
        out = einops.rearrange(out, 'b t () y x -> b t y x')
        return out

    def forward(self, x):
        return F.mse_loss(x, self.forward_ae(x))

if __name__ == '__main__':
    import sys
    import torch
    sys.path.append('../OpenSTL')
    sys.path.append('../pytorch-image-models')
    import timm
    import openstl
    import openstl.models
    import lovely_tensors
    lovely_tensors.monkey_patch()


    #  	SimVP+gSTA-Sx10 
    #
    dev = 'cuda:4' 
    mod = openstl.models.SimVP_Model([15, 1, 240, 240]).to(dev)
    class Config:
        in_shape = 15, 1, 240, 240
        stride = 1
        filter_size = 3
        layer_norm = True
        patch_size = 240


    mod = openstl.models.SimVP_Model([15, 1, 240, 240]).to(dev)
    inp = torch.randn((1, 15, 1, 240, 240), device=dev)
    out = mod(inp)
    inp, out
    

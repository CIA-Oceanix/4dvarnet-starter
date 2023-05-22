import src.models
import torch
import einops
import contrib.lorenz63


class RearrangedBilinAEPriorCost(src.models.BilinAEPriorCost):
    """
    Wrapper around the base prior cost that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, rearrange_from='b c t', rearrange_to='b t c ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

    def forward_ae(self, x):
        x = einops.rearrange(x, self.rearrange_bef)
        x = super().forward_ae(x)
        x = einops.rearrange(x, self.rearrange_aft)
        return x

class RearrangedConvLstmGradModel(src.models.ConvLstmGradModel):
    """
    Wrapper around the base grad model that allows for reshaping of the input batch
    Used to convert the lorenz timeseries into an "image" for reuse of conv2d layers
    """
    def __init__(self, rearrange_from='b c t', rearrange_to='b t c ()', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rearrange_bef = rearrange_from + ' -> ' + rearrange_to
        self.rearrange_aft = rearrange_to + ' -> ' + rearrange_from

    def reset_state(self, inp):
        inp = einops.rearrange(inp, self.rearrange_bef)
        super().reset_state(inp)

    def forward(self, x):
        x = einops.rearrange(x, self.rearrange_bef)
        x = super().forward(x)
        x = einops.rearrange(x, self.rearrange_aft)
        return x

class LitLorenz(src.models.Lit4dVarNet):
    def step(self, batch, phase="", opt_idx=None):
        return super().base_step(batch, phase)


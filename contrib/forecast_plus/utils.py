
from src.utils import get_forecast_wei
from contrib.forecast_weights import forecast_weights_reconstruction_soft_edges

def get_forecast_wei_ccut(patch_dims, ccut, **crop_kw):
    forecast_wei = get_forecast_wei(patch_dims=patch_dims, **crop_kw)
    return forecast_wei[:ccut]

def forecast_weights_reconstruction_soft_edges_ccut(*args, ccut,**kwargs):
    return forecast_weights_reconstruction_soft_edges(*args, **kwargs)[:ccut]
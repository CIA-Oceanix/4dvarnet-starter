import numpy as np

from src.utils import get_constant_crop

def get_forecast_wei_flat_expo(patch_dims, **crop_kw):
    """
    return weight for forecast reconstruction:
    exponential from 0 to 1 where there are obs,
    1 for 7 days of forecast,
    0 after

    patch_dims: dimension of the patches used
    """
    pw = get_constant_crop(patch_dims, **crop_kw)
    time_patch_weight = np.concatenate(
        (np.power(np.linspace(0, 1, (patch_dims['time'] - 1) // 2), 2),
         np.linspace(1, 1, 7),
         np.zeros((patch_dims['time'] + 1) // 2 - 7)),
        axis=0)
    final_patch_weight = time_patch_weight[:, None, None] * pw
    return final_patch_weight

# reconstruction weights

def forecast_weights_reconstruction(i, dT, dims, rec_weight):
    """
    return weights for forecast reconstruction

    i: rec weight leadtime
    dT: max time of patch_dims
    dims: rec_weight dims
    rec_weight: forecast weights (used to get the cropped center patch)
    """
    return np.concatenate(
                (np.zeros((dT // 2 + i, dims[1], dims[2])),
                np.expand_dims(rec_weight[dT//2], axis=0),
                np.zeros((dT // 2 - i, dims[1], dims[2]))),
                axis=0
            )

def forecast_weights_reconstruction_soft_edges(i, dT, dims, rec_weight, soft_edge=0.1):
    """
    return weights for forecast reconstruction

    i: rec weight leadtime
    dT: max time of patch_dims
    dims: rec_weight dims
    rec_weight: forecast weights (used to get the cropped center patch)
    soft_edge: value of edges of the reconstruction weight, relative to 1.
    """
    center_wei = rec_weight[dT//2]
    center_wei[center_wei!=0] = 1.
    center_wei[center_wei==0] = soft_edge

    return np.concatenate(
                (np.zeros((dT // 2 + i, dims[1], dims[2])),
                np.expand_dims(center_wei, axis=0),
                np.zeros((dT // 2 - i, dims[1], dims[2]))),
                axis=0
            )
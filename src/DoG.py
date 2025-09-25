import torch
import xarray as xr
import kornia
import kornia.filters as kfilts
import numpy as np

def dog_kornia(x, sigma, K):
    B, C, H, W = x.shape
    k = int(4 * sigma + 1)
    if k % 2 == 0:
        k += 1
        
    mask_bool = ~torch.isnan(x)
    
    x = torch.nan_to_num(x, nan=0.0)
    #mask_bool_tensor = torch.from_numpy(mask_bool) if isinstance(mask_bool, np.ndarray) else mask_bool
    mask_bool_tensor = mask_bool.bool()  # Convert to float (0.0 for False, 1.0 for True)
    
    # Apply the Gaussian blur with the tensor
    mask_filtered = torch.where(
        mask_bool_tensor.unsqueeze(0),
        kfilts.gaussian_blur2d(mask_bool.float(), (k, k), (sigma, sigma), separable=False),
        torch.nan
    )

    data_filtered_normalized = [kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma))]
    for i in range(1, K):
        data_filtered_normalized.append(kornia.filters.gaussian_blur2d(data_filtered_normalized[i - 1], (k, k), (sigma, sigma)))
            #torch.where(mask_bool_tensor.unsqueeze(0), kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma), separable = False), torch.nan) / (mask_filtered + 1e-6))
            
    return torch.diff(torch.stack(data_filtered_normalized, 0).squeeze(), dim = 0)

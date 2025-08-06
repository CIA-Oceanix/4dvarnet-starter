import torch
import xarray as xr

def dog_kornia(x, sigma, l3_mask):
    B, C, H, W = x.shape
    k = int(4 * sigma + 1)
    if k % 2 == 0:
        k += 1
        
    x = torch.nan_to_num(x, nan=0.0)
    #x_masked = x * l3_mask
    mask_bool = l3_mask.bool()
    mask_filtered = torch.where(mask_bool, kfilts.gaussian_blur2d(l3_mask, (k, k), (sigma, sigma), separable = False), torch.nan)

    data_filtered_normalized = []
    for i in range(K):
        data_filtered_normalized.append(torch.where(mask_bool, kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma), separable = False), torch.nan) / (mask_filtered + 1e-6))
            
    return torch.diff(torch.stack(data_filtered_normalized, 0).squeeze(), dim = 0)

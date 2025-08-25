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
    #x_masked = x * l3_mask
    #l3_mask = xr.open_dataset('/Odyssey/public/altimetry_traces/2010_2023/gridded/l3_mask.nc').l3_mask.astype('float32').values
    #print("l3_mask")
    #print(l3_mask)
    mask_bool_tensor = torch.from_numpy(mask_bool) if isinstance(mask_bool, np.ndarray) else mask_bool
    
    # Convert l3_mask (NumPy array) to a PyTorch tensor
    l3_mask_tensor = torch.from_numpy(l3_mask).float()  # Ensure the tensor is of type floa
    print('l3 mask tensor shape')
    print(l3_mask_tensor)
    
    # Apply the Gaussian blur with the tensor
    mask_filtered = torch.where(
        mask_bool_tensor[:,:,:],
        kfilts.gaussian_blur2d(l3_mask_tensor.unsqueeze(0)[:,:,:], (k, k), (sigma, sigma), separable=False),
        torch.nan
    )
    print('mask filtered[0] shape')
    print(mask_filtered[0].shape)
    print(mask_bool_tensor.shape)
    print(kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma), separable = False).shape)
    #mask_filtered = torch.where(mask_bool, kfilts.gaussian_blur2d(l3_mask, (k, k), (sigma, sigma), separable = False), torch.nan)

    data_filtered_normalized = []
    for i in range(K):
        data_filtered_normalized.append(torch.where(mask_bool_tensor.unsqueeze(0)[:,:,40:], kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma), separable = False), torch.nan) / (mask_filtered + 1e-6))
            
    return torch.diff(torch.stack(data_filtered_normalized, 0).squeeze(), dim = 0)

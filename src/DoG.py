import torch
import xarray as xr
import kornia.filters as kfilts

def dog_kornia(x, sigma):
    B, C, H, W = x.shape
    k = int(4 * sigma + 1)
    if k % 2 == 0:
        k += 1
        
    x = torch.nan_to_num(x, nan=0.0)
    #x_masked = x * l3_mask
    l3_mask = xr.open_dataset('/Odyssey/public/altimetry_traces/2010_2023/gridded/l3_mask.nc').l3_mask.astype('float32').values
    #print("l3_mask")
    #print(l3_mask)
    mask_bool = l3_mask.astype(bool)
    
    # Convert l3_mask (NumPy array) to a PyTorch tensor
    l3_mask_tensor = torch.from_numpy(l3_mask).float()  # Ensure the tensor is of type float
    
    # Apply the Gaussian blur with the tensor
    mask_filtered = torch.where(
        mask_bool,
        kfilts.gaussian_blur2d(l3_mask_tensor.unsqueeze(0), (k, k), (sigma, sigma), separable=False),
        torch.nan
    )
    #mask_filtered = torch.where(mask_bool, kfilts.gaussian_blur2d(l3_mask, (k, k), (sigma, sigma), separable = False), torch.nan)

    data_filtered_normalized = []
    for i in range(K):
        data_filtered_normalized.append(torch.where(mask_bool, kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma), separable = False), torch.nan) / (mask_filtered + 1e-6))
            
    return torch.diff(torch.stack(data_filtered_normalized, 0).squeeze(), dim = 0)

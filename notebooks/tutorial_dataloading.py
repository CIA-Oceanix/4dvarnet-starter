# %% md
"""
# Tutorial XArray to pytorch dataloading
"""

# %% md
"""
## XrDataset arguments
"""

# %%
import numpy as np
import xarray as xr

import src.data


size = 10
input_xr = xr.DataArray(
    data=np.fromfunction(lambda i,j: np.sin(i*2*np.pi/5) + np.cos(j*2*np.pi/10), (size, size)),
    coords=dict(
        d1=np.arange(size),
        d2=10 * np.arange(size),
    )
)
input_xr.plot()

patch_dims=dict(d1=7, d2=4)
strides=dict(d1=2, d2=2)
torch_dataset = src.data.XrDataset(
    input_xr, # input data
    patch_dims=patch_dims , # single item shape 
    domain_limits=None, # restrict input data domain (defaults sample from all input data)
    strides=strides, # spacing between items in each dimensions
)

coords = torch_dataset.get_coords()
def mskitem(i):
    msk = xr.zeros_like(input_xr)
    msk.loc[coords[i].coords] = 1.
    return msk

msk_da = xr.DataArray(
    np.stack([mskitem(i) for i in range(len(torch_dataset))], axis=0),
    coords=dict(
        items=np.arange(len(torch_dataset)),
        **input_xr.coords
    )
)

el_da = xr.DataArray(
    np.stack(torch_dataset, axis=0),
    coords=dict(
        items=np.arange(len(torch_dataset)),
        **{k: np.arange(v) for k,v in patch_dims.items()}
    )
)

import holoviews as hv
hv.extension('matplotlib')
def anim(da, name, climda=None):
    climda = climda if climda is not None else da
    clim = climda.pipe(lambda da: (da.quantile(0.005).item(), da.quantile(0.995).item()))
    return  (hv.Dataset(da)
            .to(hv.QuadMesh, ['d1', 'd2']).relabel(name)
            .options(cmap='RdBu',clim=clim, colorbar=True))

msk_da.sum('items').plot()

hv.output(
    anim(msk_da.rename('mask'), 'mask') 
    + anim(el_da.rename('item'), 'item'),
    fps=3, dpi=75, holomap='gif'
)

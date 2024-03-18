import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
import cartopy.crs as ccrs

data=xr.open_dataset('DMI-L4_GHRSST-SSTfnd-DMI_OI-NSEABALTIC.nc')
data2=xr.open_dataset('DMI-L4_GHRSST-SSTfnd-DMI_4DVarNet-NSEABALTIC_2021.nc')

data = data.assign_coords(lon=data2.lon)
data = data.assign_coords(lat=data2.lat)

data = data.isel(time=slice(630,994))
data2 = data2.isel(time=slice(0,364))
data = data.assign_coords(time=data2.time)

# Fidelity to Observations

## spatial analysis 
oi_fidelity = (data.analysed_sst-data2.tgt)
fourdvarnet_fidelity = (data2.analysed_sst-data2.tgt)
fig, axs = plt.subplots(2,2,figsize=(15,7.5))
np.sqrt((oi_fidelity**2).mean(dim='time',skipna=True)).plot(ax=axs[0,0],vmin=0,vmax=0.05,label='OI',cbar_kwargs={"label": "K"})
axs[0,0].set_title("RMSE(OI,Obs)")
np.sqrt((fourdvarnet_fidelity**2).mean(dim='time',skipna=True)).plot(ax=axs[0,1],vmin=0,vmax=0.05,label='4DVarNet',cbar_kwargs={"label": "K"}) 
axs[0,1].set_title("RMSE(4DVarNet,Obs)")
oi_fidelity.mean(dim='time',skipna=True).plot(ax=axs[1,0],vmin=-0.05,vmax=0.05,label='OI',cbar_kwargs={"label": "K"})
axs[1,0].set_title("MB(OI,Obs)")
fourdvarnet_fidelity.mean(dim='time',skipna=True).plot(ax=axs[1,1],vmin=-0.05,vmax=0.05,label='4DVarNet',cbar_kwargs={"label": "K"})
axs[1,1].set_title("MB(4DVarNet,Obs)")
plt.tight_layout()
plt.savefig('SST_Baltic_Comparison_Obs_spatial_2021.png')

## temporal analysis 
fig, ax = plt.subplots(1,1,figsize=(20,10))
np.sqrt((oi_fidelity**2).mean(dim=['lon','lat'],skipna=True)).plot(ax=ax,color='red',label='OI')
np.sqrt((fourdvarnet_fidelity**2).mean(dim=['lon','lat'],skipna=True)).plot(ax=ax,color='blue',label='4DVarNet')
q5_oi = np.sqrt((oi_fidelity**2).quantile(0.05,dim=['lon','lat'],skipna=True))
q95_oi = np.sqrt((oi_fidelity**2).quantile(0.95,dim=['lon','lat'],skipna=True))
q5_4dvarnet = np.sqrt((fourdvarnet_fidelity**2).quantile(0.05,dim=['lon','lat'],skipna=True))
q95_4dvarnet = np.sqrt((fourdvarnet_fidelity**2).quantile(0.95,dim=['lon','lat'],skipna=True))
ax.fill_between(q5_oi.time,q5_oi,q95_oi,color='red',alpha=0.3)
ax.fill_between(q5_oi.time,q5_4dvarnet,q95_4dvarnet,color='blue',alpha=0.3)
ax.set_ylim(0,0.25)
ax.set_ylabel("RMSE (K)")
#ax.axhline(0,linewidth=2,linestyle = '--',color='k')
plt.grid()
plt.margins(x=0)
plt.legend(loc='upper left',prop=dict(size='x-large'),frameon=False,bbox_to_anchor=(0,0.9,1,0.2),ncol=2,mode="expand")
plt.savefig('SST_Baltic_Comparison_Obs_temporal_2021.pdf')

# Persistence analysis

## spatial analysis 
obs = data2.tgt 
oi_persistence_p1 = data.analysed_sst
oi_persistence_p1 = oi_persistence_p1.assign_coords(time=oi_persistence_p1.time + np.timedelta64(1, 'D'))
oi_persistence_p1_fidelity = oi_persistence_p1-obs
fourdvarnet_persistence_p1 = data2.analysed_sst
fourdvarnet_persistence_p1 = fourdvarnet_persistence_p1.assign_coords(time=fourdvarnet_persistence_p1.time + np.timedelta64(1, 'D'))
fourdvarnet_persistence_p1_fidelity = fourdvarnet_persistence_p1-obs
fig, axs = plt.subplots(2,2,figsize=(15,7.5))
np.sqrt((oi_persistence_p1_fidelity**2).mean(dim='time',skipna=True)).plot(ax=axs[0,0],vmin=0,vmax=1,label='OI',cbar_kwargs={"label": "K"})
axs[0,0].set_title("RMSE(OI,Obs)")
np.sqrt((fourdvarnet_persistence_p1_fidelity**2).mean(dim='time',skipna=True)).plot(ax=axs[0,1],vmin=0,vmax=1,label='4DVarNet',cbar_kwargs={"label": "K"})
axs[0,1].set_title("RMSE(4DVarNet,Obs)")
oi_persistence_p1_fidelity.mean(dim='time',skipna=True).plot(ax=axs[1,0],vmin=-0.05,vmax=0.05,label='OI',cbar_kwargs={"label": "K"})
axs[1,0].set_title("MB(OI,Obs)")
fourdvarnet_persistence_p1_fidelity.mean(dim='time',skipna=True).plot(ax=axs[1,1],vmin=-0.05,vmax=0.05,label='4DVarNet',cbar_kwargs={"label": "K"})
axs[1,1].set_title("MB(4DVarNet,Obs)")
plt.tight_layout()
plt.savefig('SST_Baltic_Comparison_Obs_spatial_2021_persistence=1.png')

## temporal analysis 
fig, ax = plt.subplots(1,1,figsize=(20,10))
np.sqrt((oi_persistence_p1_fidelity**2).mean(dim=['lon','lat'],skipna=True)).plot(ax=ax,color='red',label='OI')
np.sqrt((fourdvarnet_persistence_p1_fidelity**2).mean(dim=['lon','lat'],skipna=True)).plot(ax=ax,color='blue',label='4DVarNet')
q5_oi = np.sqrt((oi_persistence_p1_fidelity**2).quantile(0.05,dim=['lon','lat'],skipna=True))
q95_oi = np.sqrt((oi_persistence_p1_fidelity**2).quantile(0.95,dim=['lon','lat'],skipna=True))
q5_4dvarnet = np.sqrt((fourdvarnet_persistence_p1_fidelity**2).quantile(0.05,dim=['lon','lat'],skipna=True))
q95_4dvarnet = np.sqrt((fourdvarnet_persistence_p1_fidelity**2).quantile(0.95,dim=['lon','lat'],skipna=True))
ax.fill_between(q5_oi.time,q5_oi,q95_oi,color='red',alpha=0.3)
ax.fill_between(q5_oi.time,q5_4dvarnet,q95_4dvarnet,color='blue',alpha=0.3)
ax.set_ylim(0,2)
ax.set_ylabel("RMSE (K)")
#ax.axhline(0,linewidth=2,linestyle = '--',color='k')
plt.grid()
plt.margins(x=0)
plt.legend(loc='upper left',prop=dict(size='x-large'),frameon=False,bbox_to_anchor=(0,0.9,1,0.2),ncol=2,mode="expand")
plt.savefig('SST_Baltic_Comparison_Obs_temporal_2021_persistence=1.pdf')

# OI vs 4DVarNet difference
fig, ax = plt.subplots(1,1,figsize=(15,7.5))
p = (data2.analysed_sst-data.analysed_sst).mean(dim=['time'],skipna=True).plot(vmin=0, vmax=.25, cbar_kwargs={"label": "K"})
ax.set_title("RMSE(OI,4DVarNet)")
plt.savefig('SST_Baltic_Comparison_DL_OI_2021.png')

"""
# 4DVarNet
p = (data2.analysed_sst).plot(aspect=2, size=3, col='time', col_wrap=6, vmin=277, vmax=297, cmap=plt.cm.plasma,transform=ccrs.PlateCarree(),subplot_kws={"projection": ccrs.Orthographic(0, 35)})

for ax in p.axs.flat:
    ax.coastlines()
    ax.gridlines()
    
plt.savefig('SST_Baltic_4DVarNet_June_2021.png')

# Obs
p = (data2.tgt).plot(aspect=2, size=3, col='time', col_wrap=6, vmin=277, vmax=297, cmap=plt.cm.plasma,transform=ccrs.PlateCarree(),subplot_kws={"projection": ccrs.Orthographic(0, 35)})

for ax in p.axs.flat:
    ax.coastlines()
    ax.gridlines()
    
plt.savefig('SST_Baltic_Obs_June_2021.png')
"""

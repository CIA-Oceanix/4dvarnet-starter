import os
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
import cartopy.crs as ccrs
import numpy as np
import sys
name_xp = sys.argv[1]

obs = xr.open_dataset('../validation_dataset/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC_validation_s0.nc',chunks={'time': 10}).isel(time=slice(10,365))
obs_valid = xr.open_dataset('../validation_dataset/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC_validation_s1.nc',chunks={'time': 10}).isel(time=slice(10,365))
oi = xr.open_dataset('../validation_dataset/DMI-L4_GHRSST-SSTfnd-DMI_OI-NSEABALTIC_2021_validation.nc',chunks={'time': 10}).isel(time=slice(10,365))
fourdvarnet = xr.open_dataset('../4DVarNet_outputs/DMI-L4_GHRSST-SSTfnd-DMI_4DVarNet-NSEABALTIC_2021_'+name_xp+'.nc',chunks={'time': 10}).isel(time=slice(10,365))
fourdvarnet = fourdvarnet.assign_coords({'lon':oi.lon.values,'lat':oi.lat.values})#,'time':oi.time.values}) 

# create new postpro directory and change current directory
os.mkdir("../postpro/result_"+name_xp)
os.chdir("../postpro/result_"+name_xp)

# Fidelity to Observations

## spatial analysis 
oi_fidelity = oi.analysed_sst-obs_valid.sea_surface_temperature
fourdvarnet_fidelity = fourdvarnet.analysed_sst-obs_valid.sea_surface_temperature
fig, axs = plt.subplots(2,2,figsize=(15,7.5))
np.sqrt((oi_fidelity**2).mean(dim='time',skipna=True)).plot(ax=axs[0,0],vmin=0,vmax=1,label='OI',cbar_kwargs={"label": "K"})
axs[0,0].set_title("RMSE(OI,Obs)")
np.sqrt((fourdvarnet_fidelity**2).mean(dim='time',skipna=True)).plot(ax=axs[0,1],vmin=0,vmax=1,label='4DVarNet',cbar_kwargs={"label": "K"}) 
axs[0,1].set_title("RMSE(4DVarNet,Obs)")
oi_fidelity.mean(dim='time',skipna=True).plot(ax=axs[1,0],vmin=-0.05,vmax=0.05,label='OI',cbar_kwargs={"label": "K"})
axs[1,0].set_title("MB(OI,Obs)")
fourdvarnet_fidelity.mean(dim='time',skipna=True).plot(ax=axs[1,1],vmin=-0.05,vmax=0.05,label='4DVarNet',cbar_kwargs={"label": "K"})
axs[1,1].set_title("MB(4DVarNet,Obs)")
plt.tight_layout()
plt.savefig('SST_Baltic_Comparison_Obs_spatial_2021.png')

# same but monthly
month_length = oi.time.dt.days_in_month
month_length
weights = (
    month_length.groupby("time.season") / month_length.groupby("time.season").sum()
)
# Test that the sum of the weights for each season is 1.0
np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))
# Calculate the weighted average
oi_fidelity_weighted = np.sqrt((oi_fidelity**2 * weights).groupby("time.season").sum(dim="time",skipna=True))
fourdvarnet_fidelity_weighted = np.sqrt((fourdvarnet_fidelity**2 * weights).groupby("time.season").sum(dim="time",skipna=True))
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(14, 12))
for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):    
    oi_fidelity_weighted.sel(season=season).plot(ax=axs[i,0],vmin=0,vmax=1,label='OI',cbar_kwargs={"label": "K"})
    fourdvarnet_fidelity_weighted.sel(season=season).plot(ax=axs[i,1],vmin=0,vmax=1,label='4DVarNet',cbar_kwargs={"label": "K"})

plt.savefig('SST_Baltic_Comparison_Obs_spatial_2021_monthly.png')

# same but monthly diff of RMSE
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
cmap_reversed = matplotlib.cm.get_cmap('RdYlGn_r')
cmap = matplotlib.cm.get_cmap('RdYlGn')
for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):    
    (oi_fidelity_weighted-fourdvarnet_fidelity_weighted).sel(season=season).plot(ax=axs[i//2,np.mod(i,2)],
                                                                                 vmin=-0.2,vmax=0.2,
                                                                                 cmap=cmap,
                                                                                 label='RMSE(OI)-RMSE(4DVarNet)',
                                                                                 cbar_kwargs={"label": "K"})

plt.savefig('SST_Baltic_Comparison_Obs_spatial_2021_diff_monthly.png')

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
ax.set_ylim(0,1.5)
ax.set_ylabel("RMSE (K)")
#ax.axhline(0,linewidth=2,linestyle = '--',color='k')
plt.grid()
plt.margins(x=0)
plt.legend(loc='upper left',prop=dict(size='x-large'),frameon=False,bbox_to_anchor=(0,0.9,1,0.2),ncol=2,mode="expand")
plt.savefig('SST_Baltic_Comparison_Obs_temporal_2021.pdf')

# OI vs 4DVarNet difference
fig, ax = plt.subplots(1,1,figsize=(15,7.5))
(oi.analysed_sst-fourdvarnet.analysed_sst).mean(dim=['time'],skipna=True).plot(vmin=0, vmax=.25, cbar_kwargs={"label": "K"})
ax.set_title("RMSE(OI,4DVarNet)")
plt.savefig('SST_Baltic_Comparison_DL_OI_2021.png')

# same but monthly
oi_weighted = (oi * weights).groupby("time.season").sum(dim="time",skipna=True)
fourdvarnet_weighted = (fourdvarnet * weights).groupby("time.season").sum(dim="time",skipna=True)
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
for i, season in enumerate(("DJF", "MAM", "JJA", "SON")):    
    (oi_weighted.analysed_sst-fourdvarnet_weighted.analysed_sst).sel(season=season).plot(ax=axs[i//2,np.mod(i,2)],vmin=-0.1,vmax=0.1,cbar_kwargs={"label": "K"})
    
    
plt.savefig('SST_Baltic_Comparison_DL_OI_2021_monthly.png')




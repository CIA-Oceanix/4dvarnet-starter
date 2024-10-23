import xarray as xr
import src
import numpy as np
import contrib

def load_data(path_obs="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
              path_tgt="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc") :

    
    inp = xr.open_dataset(path_obs).rename_vars({"sea_surface_temperature":"input"}).transpose('time', 'lat', 'lon')
    tgt = xr.open_dataset(path_tgt).rename_vars({"sea_surface_temperature":"tgt"}).transpose('time', 'lat', 'lon')  
    data = inp
    data['tgt'] = tgt.tgt
    data = data[[*contrib.dmi.data.TrainingItem._fields]]
    
    return data

def load_data_wcoarse(path_obs="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                      path_tgt="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                      path_coarse="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc") :
    
    
    inp = xr.open_dataset(path_obs).rename_vars({"sea_surface_temperature":"input"}).transpose('time', 'lat', 'lon')
    tgt = xr.open_dataset(path_tgt).rename_vars({"sea_surface_temperature":"tgt"}).transpose('time', 'lat', 'lon')
    coarse = xr.open_dataset(path_coarse).rename_vars({"analysed_sst_LR":"coarse"}).transpose('time', 'lat', 'lon')
            
    data = inp
    data['tgt'] = tgt.tgt
    data['coarse'] = coarse.coarse.drop_vars("mask")
    data = data[[*contrib.dmi.data.TrainingItem_wcoarse._fields]]
    if 'mask' in list(data.keys()):
        data = data.drop_vars("mask")
        
    return data

def load_data_wgeo(path_obs="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                   path_tgt="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                   path_oi="/DATASET/mbeauchamp/DMI/DMI-L4_GHRSST-SSTfnd-DMI_OI-NSEABALTIC.nc",
                   path_topo="/DATASET/mbeauchamp/DMI/DMI-TOPO_NSEABALTIC.nc",
                   path_fgstd="/DATASET/mbeauchamp/DMI/DMI-FGSTD_NSEABALTIC.nc") :

    mask = xr.open_dataset(path_oi)
    topo = xr.open_dataset(path_topo)
    fg_std = xr.open_dataset(path_fgstd)

    inp = xr.open_dataset(path_obs).rename_vars({"sea_surface_temperature":"input"}).transpose('time', 'lat', 'lon')
    tgt = xr.open_dataset(path_tgt).rename_vars({"sea_surface_temperature":"tgt"}).transpose('time', 'lat', 'lon')

    data = inp
    data['tgt'] = tgt.tgt   
    data = data.update({'latv':(('lat','lon'),data.lat.broadcast_like(data.tgt[0]).data),
                        'lonv':(('lat','lon'),data.lon.broadcast_like(data.tgt[0]).data),
                        'land_mask':(('lat','lon'),np.isnan(mask.analysed_sst[0]).values.astype(int)),
                        'topo':(('lat','lon'),np.log(-1.*topo.topo.data+1)),
                        'fg_std':(('lat','lon'),fg_std.sat_var.data)})
    data = data[[*contrib.dmi.data.TrainingItem_wgeo._fields]]
    if 'mask' in list(data.keys()):
        data = data.drop_vars("mask")
        
    return data

def load_data_wcoarse_wgeo(path_obs="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                   path_tgt="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                   path_oi="/DATASET/mbeauchamp/DMI/DMI-L4_GHRSST-SSTfnd-DMI_OI-NSEABALTIC.nc",
                   path_coarse="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                   path_topo="/DATASET/mbeauchamp/DMI/DMI-TOPO_NSEABALTIC.nc",
                   path_fgstd="/DATASET/mbeauchamp/DMI/DMI-FGSTD_NSEABALTIC.nc"):

    mask = xr.open_dataset(path_oi)
    topo = xr.open_dataset(path_topo)
    fg_std = xr.open_dataset(path_fgstd)

    inp = xr.open_dataset(path_obs).rename_vars({"sea_surface_temperature":"input"}).transpose('time', 'lat', 'lon')
    tgt = xr.open_dataset(path_tgt).rename_vars({"sea_surface_temperature":"tgt"}).transpose('time', 'lat', 'lon')
    coarse = xr.open_dataset(path_coarse).rename_vars({"analysed_sst_LR":"coarse"}).transpose('time', 'lat', 'lon')
            
    data = inp
    data['tgt'] = tgt.tgt
    data['coarse'] = coarse.coarse.drop_vars("mask")   
    data = data.update({'latv':(('lat','lon'),data.lat.broadcast_like(data.tgt[0]).data),
                        'lonv':(('lat','lon'),data.lon.broadcast_like(data.tgt[0]).data),
                        'land_mask':(('lat','lon'),np.isnan(mask.analysed_sst[0]).values.astype(int)),
                        'topo':(('lat','lon'),np.log(-1.*topo.topo.data+1)),
                        'fg_std':(('lat','lon'),fg_std.sat_var.data)})
    data = data[[*contrib.dmi.data.TrainingItem_wcoarse_wgeo._fields]]
    #data = data.chunk(chunks={"time": -1})
    if 'mask' in list(data.keys()):
        data = data.drop_vars("mask")
        
    return data
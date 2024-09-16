import xarray as xr
import src
import numpy as np
import contrib

def load_data(path_obs="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
              path_tgt="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc") :
    data = xr.merge([
             xr.open_dataset(path_obs,chunks={'time': 10}).load().assign(
                 input=lambda ds: ds.sea_surface_temperature
             ),
             xr.open_dataset(path_tgt,chunks={'time': 10}).load().assign(
                 tgt=lambda ds: ds.analysed_sst#sea_surface_temperature
             )]
           ,compat='override')[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon')#.to_array().load()
    return data

def load_data_wcoarse(path_obs="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                      path_tgt="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                      path_coarse="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc") :
    data = xr.merge([
             xr.open_dataset(path_obs,chunks={'time': 10}).load().assign(
                 input=lambda ds: ds.sea_surface_temperature
             ),
             xr.open_dataset(path_tgt,chunks={'time': 10}).load().assign(
                 tgt=lambda ds: ds.sea_surface_temperature
             ),
             xr.open_dataset(path_coarse,chunks={'time': 10}).load().assign(
                 coarse=lambda ds: ds.analysed_sst_LR
             )]
           ,compat='override')[[*contrib.dmi.data.TrainingItem_wcoarse._fields]].transpose('time', 'lat', 'lon')

    data = data.update({'input':(('time','lat','lon'),data.input.data-data.coarse.data),
                        'tgt':(('time','lat','lon'),data.tgt.data-data.coarse.data)})

    return data

def load_data_wgeo(path_obs="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                   path_tgt="/DATASET/mbeauchamp/DMI/DMI-L3S_GHRSST-SSTsubskin-night_SST_UHR_NRT-NSEABALTIC.nc",
                   path_oi="/DATASET/mbeauchamp/DMI/DMI-L4_GHRSST-SSTfnd-DMI_OI-NSEABALTIC.nc",
                   path_topo="/DATASET/mbeauchamp/DMI/DMI-TOPO_NSEABALTIC.nc",
                   path_fgstd="/DATASET/mbeauchamp/DMI/DMI-FGSTD_NSEABALTIC.nc") :

    mask = xr.open_dataset(path_oi)
    topo = xr.open_dataset(path_topo)
    fg_std = xr.open_dataset(path_fgstd)

    data = xr.merge([
             xr.open_dataset(path_obs).rename_vars({"sea_surface_temperature":"input"}),
             xr.open_dataset(path_tgt).rename_vars({"sea_surface_temperature":"tgt"})]
           ,compat='override')[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon')#.to_array().load()
    data = data.update({'latv':(('lat','lon'),data.lat.broadcast_like(data.tgt[0]).data),
                        'lonv':(('lat','lon'),data.lon.broadcast_like(data.tgt[0]).data),
                        'mask':(('lat','lon'),np.isnan(mask.analysed_sst[0]).values.astype(int)),
                        'topo':(('lat','lon'),np.nan_to_num(topo.bathymetry.data)),
                        'fg_std':(('lat','lon'),fg_std.std.data)})
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

    data = xr.merge([
             xr.open_dataset(path_obs).rename_vars({"sea_surface_temperature":"input"}),
             xr.open_dataset(path_tgt).rename_vars({"sea_surface_temperature":"tgt"}),
             xr.open_dataset(path_coarse).rename_vars({"analysed_sst_LR":"coarse"})
             ],compat='override')[[*contrib.dmi.data.TrainingItem_wcoarse._fields]].transpose('time', 'lat', 'lon')#.to_array().load()

    data = data.update({'latv':(('lat','lon'),data.lat.broadcast_like(data.tgt[0]).data),
                        'lonv':(('lat','lon'),data.lon.broadcast_like(data.tgt[0]).data),
                        'mask':(('lat','lon'),np.isnan(mask.analysed_sst[0]).values.astype(int)),
                        'topo':(('lat','lon'),np.log(-1.*topo.topo.data+1)),
                        'fg_std':(('lat','lon'),fg_std.sat_var.data)})
    data = data.update({'input':(('time','lat','lon'),data.input.data-data.coarse.data),
                        'tgt':(('time','lat','lon'),data.tgt.data-data.coarse.data)})

    return data


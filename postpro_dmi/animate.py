import datetime
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import colors as cl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shapely
from shapely import wkt
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cv2
import matplotlib.animation as animation
import pandas as pd
from celluloid import Camera

def plot(ax,lon,lat,data,title,cmap,norm,extent=[-65,-55,30,40],colorbar=True,orientation="horizontal"):
    ax.set_extent(list(extent))
    im=ax.pcolormesh(lon, lat, data, cmap=cmap,norm=norm,
                          transform= ccrs.PlateCarree(central_longitude=0.0))
    if colorbar==True:
        clb = plt.colorbar(im, orientation=orientation, extend='both', pad=0.1, ax=ax)
    gl = ax.gridlines(alpha=0.5,draw_labels=True)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabels_bottom = False
    gl.ylabels_right = False
    gl.xlabel_style = {'fontsize': 10, 'rotation' : 45}
    gl.ylabel_style = {'fontsize': 10}
    ax.coastlines(resolution='50m')
    return im
    
def animate_maps(gt, pred, lon, lat, date, resfile):

    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(121,projection=ccrs.PlateCarree(central_longitude=0.0))
    ax2 = fig.add_subplot(122,projection=ccrs.PlateCarree(central_longitude=0.0))
    vmax = 298.
    vmin = 276.

    # create the colorbar
    cbar_ax = fig.add_axes([0.1, 0.15, 0.8, 0.03])
    cm = plt.cm.coolwarm
    norm = cl.Normalize(vmin=vmin, vmax=vmax)
    extent = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    plt.subplots_adjust(hspace=0.5)

    camera = Camera(fig)

    for i in range(len(pred)):
        print(i)
        ax1.text(0.2, 1.2, 'SST (Obs) '+ np.datetime_as_string(date[i], unit='D'),
                 transform=ax1.transAxes,fontsize='x-large')
        ax2.text(0.2, 1.2, '4DVarNet-SST '+np.datetime_as_string(date[i], unit='D'),
                 transform=ax2.transAxes,fontsize='x-large')
        im1 = plot(ax1,lon,lat,gt[i],'SST (Obs) '+np.datetime_as_string(date[i], unit='D'),
             extent=extent,cmap=cm,norm=norm,colorbar=False)
        im2 = plot(ax2,lon,lat,pred[i],'4DVarNet-SST '+np.datetime_as_string(date[i], unit='D'),
             extent=extent,cmap=cm,norm=norm,colorbar=False)
        sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', label='(K)', pad=3.0)
        
        camera.snap()

    animation = camera.animate()
    animation.save(resfile)

    plt.close()
    

data=xr.open_dataset('DMI-L4_GHRSST-SSTfnd-DMI_4DVarNet-NSEABALTIC_2021.nc')
#data = data.isel(time=slice(7,37))

lon=data.lon.data
lat=data.lat.data
time=data.time.data

animate_maps(data.tgt.data,
             data.analysed_sst.data,
             lon, lat, date=time,resfile='anim.mp4')



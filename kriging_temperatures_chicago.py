# compare this with my spatial interpolation: 
# https://github.com/VincentGranville/Statistical-Optimization/blob/main/interpol.py
# This kriging oversmooth the temperatures, compared to my method

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors  
from matplotlib import cm # color maps
import osmnx as ox
import pandas as pd
import glob
from pykrige.ok import OrdinaryKriging
from pykrige.kriging_tools import write_asc_grid
import pykrige.kriging_tools as kt
from matplotlib.colors import LinearSegmentedColormap

city = ox.geocode_to_gdf('Chicago, IL')
city.to_file("il-chicago.shp")

# dataset:  https://raw.githubusercontent.com/VincentGranville/Statistical-Optimization/main/sensors.csv
data = pd.read_csv(
     'sensors.csv',
     delim_whitespace=False, header=None,
     names=["Lat", "Lon", "Z"])

lons=np.array(data['Lon']) 
lats=np.array(data['Lat']) 
zdata=np.array(data['Z'])

import geopandas as gpd
Chicago_Boundary_Shapefile = 'il-chicago.shp'
boundary = gpd.read_file(Chicago_Boundary_Shapefile)

# get the boundary of Chicago 
xmin, ymin, xmax, ymax = boundary.total_bounds

xmin = xmin-0.06
xmax = xmax+0.05
ymin = ymin-0.01
ymax = ymax+0.01
grid_lon = np.linspace(xmin, xmax, 100)
grid_lat = np.linspace(ymin, ymax, 100)

#------
# ordinary kriging

OK = OrdinaryKriging(lons, lats, zdata, variogram_model='gaussian', verbose=True, enable_plotting=False,nlags=20)
z1, ss1 = OK.execute('grid', grid_lon, grid_lat)
print (z1)

#-------
# plots

xintrp, yintrp = np.meshgrid(grid_lon, grid_lat) 
plt.rcParams['axes.linewidth'] = 0.3
fig, ax = plt.subplots(figsize=(8,6))

contour = plt.contourf(xintrp, yintrp, z1,len(z1),cmap=plt.cm.jet,alpha = 0.8)
cbar = plt.colorbar(contour)
cbar.ax.tick_params(width=0.1) 
cbar.ax.tick_params(length=2)
cbar.ax.tick_params(labelsize=7) 

boundary.plot(ax=ax, color='white', alpha = 0.2, linewidth=0.5, edgecolor='black', zorder = 5)
npts = len(lons)

plt.scatter(lons, lats,marker='o',c='b',s=8)  
plt.xticks(fontsize = 7) 
plt.yticks(fontsize = 7)
plt.show()

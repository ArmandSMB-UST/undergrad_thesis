# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:43:43 2022

@author: ArmandSMB
"""
import numpy as np
import os

landfall_mode = False
nearest_point_mode = True
nearest_point_threshold = 400

# Data Resolution parameters
dt = 3                                                                          #time interval

# Parameters
column_names = ["ID", "Name", "Timestamp", "Latitude", "Longitude",
                "Time", "Coriolis Parameter", "Pressure", "Max Wind", 
                "Wind at 10m", "Translational Velocity", "Distance From Manila",
                "Rainfall"]

# Manila coordinates
latManila = 14.6436
longManila = 121.0437

# Data
textFile = "raw/bst_2010_to_2020.txt"
csvFile = "raw/Science_Garden_data_2010_to_2021_" + str(dt) + "h_resolution.csv"

# Earth parameters
radius_equator = 6378.137
radius_polar = 6356.752

from mpl_toolkits.basemap import Basemap

# For plotting
mapForTrackALL = Basemap(llcrnrlon= 112, llcrnrlat=3, urcrnrlon=145.,
                    urcrnrlat=22., resolution='i', projection='tmerc',
                    lat_0 = 12, lon_0 = 120)
mapForTrack2 = Basemap(llcrnrlon= 113.5, llcrnrlat=10.5, urcrnrlon=129,
                    urcrnrlat=21.1, resolution ='i', projection='tmerc',
                    lat_0 = 14.7, lon_0 = 121)
mapForTrack3 = Basemap(llcrnrlon= 114, llcrnrlat=10.5, urcrnrlon=127,
                    urcrnrlat=19, resolution ='f', projection='tmerc',
                    lat_0 = 14.7, lon_0 = 121)

# Original llcrnrlat = 4.7
# =============================================================================
# mapForTrack = Basemap(llcrnrlon= 116, llcrnrlat=10.0, urcrnrlon=128.,
#                     urcrnrlat=19.6, resolution ='i', projection='tmerc',
#                     lat_0 = 12, lon_0 = 120)
# =============================================================================
mapForTrack = Basemap(llcrnrlon= 120.5, llcrnrlat=14, urcrnrlon=121.5,
                    urcrnrlat=15, resolution ='i', projection='tmerc',
                    lat_0 = 12, lon_0 = 120)

def save_coastal_data(path):
    m = Basemap(llcrnrlon= 116, llcrnrlat=4.7, urcrnrlon=128.,
                    urcrnrlat=19.6, resolution ='l', projection='tmerc',
                    lat_0 = 12, lon_0 = 120)

    coast = m.drawcoastlines()

    coordinates = np.vstack(coast.get_segments())
    lons,lats = m(coordinates[:,0],coordinates[:,1],inverse=True)

    np.savez(os.path.join(path,'coastal_basemap_data.npz'),a=lats, b=lons)

if __name__ == '__main__':
    save_coastal_data('../raw')
    
elif __name__ != '__main__':
    # Coast coordinates
    coastCoords = np.load('raw/coastal_basemap_data.npz')
    coastLats = coastCoords['a']
    coastLongs = coastCoords['b']
    coastLats = np.round_(coastLats, decimals = 2)
    coastLongs = np.round_(coastLongs, decimals = 2)
    coastCoordinates = {'lat': coastLats, 'long': coastLongs}

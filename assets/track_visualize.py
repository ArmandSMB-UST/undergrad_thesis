# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:09:21 2022

@author: ArmandSMB
"""
#from mayavi import mlab
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def trackVisualize(dataframe, ids, showLegend = False):
    import assets.config as config
    import assets.intense_maths as mafs
    # input trainX dataset
    base_map = plt.figure(figsize=(8, 12))
    base_map = config.mapForTrackALL
    base_map.drawmapboundary(fill_color='#87cdf6')
    base_map.fillcontinents(color='#20c506',lake_color='#87cdf6')
    base_map.drawcoastlines()
    for i in range(len(ids)):
        long = [val for j, val in enumerate(dataframe['Longitude']) if ids[i] == dataframe['ID'][j]]
        lat = [mafs.corPamToLat(val) for j, val in enumerate(dataframe['Coriolis Parameter'])
               if ids[i] == dataframe['ID'][j]]
        long, lat = base_map(long, lat)
        base_map.plot(long, lat, color='#B90F1B', linewidth=0.75, label = ids[i])
        base_map.scatter(long, lat, marker='.', color='#B90F1B')
    if showLegend == True:
        plt.legend()

if __name__ == '__main__':    
    df = pd.read_csv("../processed/nearest-as-reference.csv", index_col = [0])
    
    timeseries = np.arange(0, 51, 3)

    listFloater = lambda lst: np.array([float(val) for val in lst[1:len(lst) - 1].split(',')])
    
    plt.figure(figsize=(12,12))
    
    rainfall_total = np.zeros(int(2 * 24 / 3) + 1)
    for i in range(len(df['Rainfall'])):    
        rainfall = listFloater(df['Rainfall'][i])
        rainfall_total = [rainfall_total[i] + rainfall[i] for i in range(len(rainfall))]
        
        r = listFloater(df['Distance From Manila'][i])
        plt.subplot(322)
        plt.title(label = "r v. t")
        plt.scatter(timeseries, r)
        
        pres = listFloater(df['Pressure'][i])
        plt.subplot(323)
        plt.title(label = "pres v. t")
        plt.scatter(timeseries, pres)
        
        maxWind = listFloater(df['Max Wind'][i])
        plt.subplot(324)
        plt.title(label = "maxWind v. t")
        plt.scatter(timeseries, maxWind)
        
        wind10m = listFloater(df['Wind at 10m'][i])
        plt.subplot(325)
        plt.title(label = "wind at 10m v. t")
        plt.scatter(timeseries, wind10m)
        
        vel = listFloater(df['Translational Velocity'][i])
        plt.subplot(326)
        plt.title(label = "vel v. t")
        plt.scatter(timeseries, vel)
    
    rainfall_total = [val/sum(i for i in rainfall_total) for val in rainfall_total]
    
    plt.subplot(321)
    plt.title(label = "Accumulative rf")
    plt.xlim(0, 48)
    plt.bar(timeseries[1:], rainfall_total[1:], tick_label = [i for i in timeseries[1:]],
            width = -3.0, align = 'edge', edgecolor = '#000000')
    
    plt.show()
# =============================================================================
#     # Earth Parameters
#     r_polar = 6356.752
#     r_equator = 6378.137
#     pi = np.pi
#     cos = np.cos
#     sin = np.sin
#     
#     ups, phi = np.mgrid[-pi/2:pi/2:101j, -pi:pi:101j]
#     
#     x = r_equator*cos(ups)*cos(phi)
#     y = r_equator*cos(ups)*sin(phi)
#     z = r_polar*sin(ups)
#     
#     mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
#     mlab.clf()
#     
#     mlab.mesh(x , y , z, colormap="ocean")
#     
#     # Sample Track Points
#     ups_in = [9.3, 10., 10.7, 11.7, 12.4, 13.4, 14.4, 15.2, 15.9, 16.7, 17.6,
#            18.7, 19.3, 19.9, 19.5, 19.1, 18.9, 18.8, 18.6, 18.5, 18.8, 19.3,
#            19.8, 20.3, 21. , 21.1, 21.1, 21.1, 20.7, 20.6, 20.4, 20.3, 20.1,
#            19.8, 19.6, 19.1]
#     ups_in = np.multiply(ups_in, pi/180)
#     
#     phi_in = [129.1, 129.1, 128.5, 127.8, 127.4, 126.7, 125.6, 124.9, 124.5,
#            124.2, 123.7, 122.6, 121. , 119.7, 118.7, 117.5, 116.6, 116.1,
#            115.5, 114.8, 114.4, 114. , 113.5, 113.1, 112.6, 111.8, 110.9,
#            109.7, 109. , 108.2, 107.2, 106.4, 105.2, 104.3, 103.9, 103.3]
#     phi_in = np.multiply(phi_in, pi/180)
#     
#     xx = r_equator*cos(ups_in)*cos(phi_in)
#     yy = r_equator*cos(ups_in)*sin(phi_in)
#     zz = r_polar*sin(ups_in)
#     mlab.points3d(xx, yy, zz, scale_factor=100)
#     
#     mlab.show()
# =============================================================================

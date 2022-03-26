# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 21:09:21 2022

@author: ArmandSMB
"""
from mayavi import mlab
import numpy as np

if __name__ == '__main__':
    # Earth Parameters
    r_polar = 6356.752
    r_equator = 6378.137
    pi = np.pi
    cos = np.cos
    sin = np.sin
    
    ups, phi = np.mgrid[-pi/2:pi/2:101j, -pi:pi:101j]
    
    x = r_equator*cos(ups)*cos(phi)
    y = r_equator*cos(ups)*sin(phi)
    z = r_polar*sin(ups)
    
    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 400))
    mlab.clf()
    
    mlab.mesh(x , y , z, colormap="ocean")
    
    # Sample Track Points
    ups_in = [9.3, 10., 10.7, 11.7, 12.4, 13.4, 14.4, 15.2, 15.9, 16.7, 17.6,
           18.7, 19.3, 19.9, 19.5, 19.1, 18.9, 18.8, 18.6, 18.5, 18.8, 19.3,
           19.8, 20.3, 21. , 21.1, 21.1, 21.1, 20.7, 20.6, 20.4, 20.3, 20.1,
           19.8, 19.6, 19.1]
    ups_in = np.multiply(ups_in, pi/180)
    
    phi_in = [129.1, 129.1, 128.5, 127.8, 127.4, 126.7, 125.6, 124.9, 124.5,
           124.2, 123.7, 122.6, 121. , 119.7, 118.7, 117.5, 116.6, 116.1,
           115.5, 114.8, 114.4, 114. , 113.5, 113.1, 112.6, 111.8, 110.9,
           109.7, 109. , 108.2, 107.2, 106.4, 105.2, 104.3, 103.9, 103.3]
    phi_in = np.multiply(phi_in, pi/180)
    
    xx = r_equator*cos(ups_in)*cos(phi_in)
    yy = r_equator*cos(ups_in)*sin(phi_in)
    zz = r_polar*sin(ups_in)
    mlab.points3d(xx, yy, zz, scale_factor=100)
    
    mlab.show()
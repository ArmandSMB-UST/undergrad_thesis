# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 17:19:43 2022

@author: ArmandSMB
"""
import math
import assets.config as config
import numpy as np

# Variable simplification for easy reading
t = config.dt
dt = t
cos = math.cos
sin = math.sin
sqrt = math.sqrt
sigma_1 = config.radius_equator
sigma_2 = config.radius_polar
lat_Manila = config.latManila
long_Manila = config.longManila

radToDeg = lambda rad: rad * 180 / math.pi
degToRad = lambda degree: degree * math.pi / 180
lat_Manila = degToRad(lat_Manila)
long_Manila = degToRad(long_Manila)

def interpolateParameter(parameter_list):
    n = 6 / dt
    parameter = np.array([])
    param = 0
    param_0 = 0
    for i in range(len(parameter_list)):
        param = float(parameter_list[i])
        if i > 0:
            param_0 = float(parameter_list[i-1])
            parameter = np.append(parameter, param_0)
            
            param_avg = (param + param_0) / n
            parameter = np.append(parameter, param_avg)
    parameter = np.append(parameter, param)
    return parameter

def solveDistanceFromManila(lat, long):
    ins = vars()
    for input_name in ins:
        val = degToRad(ins[input_name])
        ins[input_name] = val
    r = (sigma_1 * sqrt( (cos(ins['lat']) * (ins['long'] - long_Manila))**2
        + (sin(ins['lat'])**2 + ((sigma_2/sigma_1) * cos(ins['lat']))**2)
         * (ins['lat'] - lat_Manila)**2 ))
    return r
    
def solveArcLength(lat_0, lat, long_0, long):
    '''lat = upsilon, long = phi'''
    ins = vars()
    for input_name in ins:
        val = degToRad(ins[input_name])
        ins[input_name] = val
    s = (sigma_1 * sqrt( (cos(ins['lat']) * (ins['long'] - ins['long_0']))**2
        + (sin(ins['lat'])**2 + ((sigma_2/sigma_1) * cos(ins['lat']))**2)
         * (ins['lat']-ins['lat_0']) **2 ) )
    return s
    
def solveTranslationalVelocity(arc_length):
    v = arc_length / t
    return v

def interpolateCoords(ups_list, phi_list):
    n = 6 / dt
    solveCoord = lambda coord_0, coord, dCoord: ((coord_0 - dCoord) if coord_0 > coord
                                        else (coord_0 + dCoord))
    lat = 0
    long = 0
    lat_0 = 0
    long_0 = 0
    latitude = np.array([])
    longitude = np.array([])
    for i in range(len(ups_list)):
        lat = ups_list[i]
        long = phi_list[i]
        if i == 0:
            s = 0
        
        elif i > 0 and i < len(ups_list):
            lat_0 = ups_list[i-1]
            latitude = np.append(latitude, lat_0)
            
            long_0 = phi_list[i-1]
            s = solveArcLength(lat_0, lat, long_0, long) / n
            longitude = np.append(longitude, long_0)
            
            lat = degToRad(lat)
            long = degToRad(long)
            lat_0 = degToRad(lat_0)
            long_0 = degToRad(long_0)
            if lat == lat_0:                                                    #from x_0 to x
                change_in_phi = (s / (cos(lat) * sigma_1))
                long = solveCoord(long_0, long, change_in_phi)
                latitude = np.append(latitude, round(radToDeg(lat), 2))
                longitude = np.append(longitude, round(radToDeg(long), 2))
                
            elif long == long_0:                                                #from x to x_0
                lat_0 = lat - (s / (sigma_1 * sqrt( sin(lat)**2 + 
                                   ((sigma_2 * cos(lat)) / sigma_1)**2 )))
                latitude = np.append(latitude, round(radToDeg(lat_0), 2))
                longitude = np.append(longitude, round(radToDeg(long_0), 2))

            else:
                lat = (lat + lat_0) / n                                         #Assumed avg movement
                latitude = np.append(latitude, round(radToDeg(lat), 2))
                
                change_in_phi = ( sqrt( s**2 - ( (lat-lat_0)**2 * 
                              ((sigma_1 * sin(lat))**2 + (sigma_2 * cos(lat))**2 )) )
                                       / (sigma_1 * cos(lat)) )
                long = solveCoord(long_0, long, change_in_phi)
                longitude = np.append(longitude, round(radToDeg(long), 2))

    # Extract final value
    latitude = np.append(latitude, ups_list[len(ups_list)-1])
    longitude = np.append(longitude, phi_list[len(phi_list)-1])
    
    return latitude, longitude
        
if __name__ == '__main__':
    import config
    dt = 3
    pressure = ['1008', '1008', '1006', '1008', '1006', '1006', '1004', '1004',
       '1002', '1002', '1002', '1004', '1002', '1002', '1000', '1000',
       '998', '998', '996', '994', '994', '994', '992', '990', '990',
       '990', '990', '990', '990', '985', '985', '990', '990', '992',
       '996', '996']
    print(len(pressure))
    pressure = interpolateParameter(pressure)
    print(len(pressure))
    
# =============================================================================
#     ups_in = [9.3, 10., 10.7, 11.7, 12.4, 13.4, 14.4, 15.2, 15.9, 16.7, 17.6,
#            18.7, 19.3, 19.9, 19.5, 19.1, 18.9, 18.8, 18.6, 18.5, 18.8, 19.3,
#            19.8, 20.3, 21. , 21.1, 21.1, 21.1, 20.7, 20.6, 20.4, 20.3, 20.1,
#            19.8, 19.6, 19.1]
#     phi_in = [129.1, 129.1, 128.5, 127.8, 127.4, 126.7, 125.6, 124.9, 124.5,
#            124.2, 123.7, 122.6, 121. , 119.7, 118.7, 117.5, 116.6, 116.1,
#            115.5, 114.8, 114.4, 114. , 113.5, 113.1, 112.6, 111.8, 110.9,
#            109.7, 109. , 108.2, 107.2, 106.4, 105.2, 104.3, 103.9, 103.3]
#     
#     latitude, longitude = interpolateCoords(ups_in, phi_in)
#     print(latitude)
#     print("Input array length: %g, Output array length: %g" %(len(ups_in), len(latitude)))
#     print(longitude)
#     print("Input array length: %g, Output array length: %g" %(len(phi_in), len(longitude)))
# =============================================================================

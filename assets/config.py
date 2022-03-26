# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:43:43 2022

@author: ArmandSMB
"""
# Data Resolution parameters
dt = 3                                                                          #time interval

# Manila coordinates
latManila = 14.6042
longManila = 120.9822

# Data
textFile = "raw/bst_2010_to_2020.txt"
csvFile = "raw/Manila_data_2010_to_2021_" + str(dt) + "h_resolution.csv"

# Earth parameters
radius_equator = 6378.137
radius_polar = 6356.752
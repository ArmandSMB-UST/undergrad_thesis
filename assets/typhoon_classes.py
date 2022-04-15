# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 14:33:54 2022

@author: ArmandSMB
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import assets.csv_operations as csv_ops
import assets.config as config
import assets.intense_maths as mafs

# Variable simplification for easy reading
dt = config.dt

class header_data:
    """""""""""
Header Line: AAAAA      BBBB     CCC       DDDD    EEEE     F       G       HHHHHHHHHHHHHHHHHHHH    IIIIIIII
            Indicator  IntlID  DataLines Tropical  BBBB    If in   diff           Name             Latest Rev. Date
                                        Cyclone ID         Japan   b/w 
                                                                  last pt
                                                              & time analysis
    """""""""  
    def __init__(self, internationalID, dataLine, typhoonName):
        self.internationalID = internationalID
        self.dataLine = dataLine
        self.typhoonName = typhoonName
    
    def getHeaders(self, file):
        self.dataFile(file)
        self.getHeaderLines()
        for element in self.trimmed_headers:
            self.internationalID.append(element[0])
            self.dataLine.append(element[1])
            self.typhoonName.append(element[2])
            
    def dataFile(self, file):
        self.file = file
        
    def countTotalLines(self):
        with open(self.file,"r") as myFile:
            total_lines = sum(1 for _ in myFile)
        return total_lines
            
    def getHeaderLines(self):
        self.full_headers = []
        self.trimmed_headers = []
        count = 0
        typhoon_count = 0
        total_lines = self.countTotalLines()
        with open(self.file) as myFile:
            for line in myFile:
                line_read = line
                while count < (total_lines + 1):                                #Idk, wonky w/o this
                    space_count = 1
                    if line:
                        line_read.split()                                       #remove whitespaces
                        line_read = " ".join(line_read.split())   
                        line_list = list(line_read.split(" "))                  #convert line to list
                        for i in line_read:
                            if i == " ":
                                space_count += 1
                                if (space_count == 8) and (len(line_list) == 9):#header lines spaces
                                    typhoon_count += 1
                                    line_to_save = [line_list[1], line_list[2],
                                                    line_list[7]]
                                    self.full_headers.append(line_read)
                                    self.trimmed_headers.append(line_to_save)
                        count += 1
                    line_read = myFile.readline()                               #Prepares next line
        return self.full_headers, self.trimmed_headers
    
# ====================================================================================================================
#    
# ====================================================================================================================
     
class data_lines(header_data):    
    """""""""""
Data Line: AAAAAAAA    BBB      C    DDD  EEEE   FFFF   GGG      HIIII   JJJJ     KLLLL  MMMM         P
           yymmddhh Indicator Grade  Lat. Long.  hPa  sus.wind  dir-long short  dir-long short    landfall JP
                                                                 rad of >50kt    rad of >30kt
    """""""""
    def __init__(self, *args, **kwargs):   
        super().__init__(*args, **kwargs)                                       #inherits __init__ attri of header_data
     
    def intializeAttriPreLandfallCheck(self):
        self.timestamp = np.array([])
        self.latitude = np.array([])
        self.longitude = np.array([])
        self.pressure = np.array([])
        self.maxWind = np.array([])
        self.displacement = np.array([])
        self.velocity = np.array([])
        self.windAt10m = np.array([])                                           # in Manila
        self.distManila = np.array([])
        for element in self.data_lines:
            if int(element[0][6:]) % 6 == 0:                                    # check timestamp interval
                self.timestamp = np.append(self.timestamp, element[0])
                self.latitude = np.append(self.latitude, float(element[3])*0.1)     # conversion unit
                self.longitude = np.append(self.longitude, float(element[4])*0.1)
                self.pressure = np.append(self.pressure, element[5])
                self.maxWind = np.append(self.maxWind, int(element[6]))
        if dt != 6:
            self.timestamp = csv_ops.extractTimestamp(config.csvFile, self.timestamp)
            self.latitude, self.longitude = mafs.interpolateCoords(self.latitude, self.longitude)
            self.pressure = mafs.interpolateParameter(self.pressure)
            self.maxWind = mafs.interpolateParameter(self.maxWind)

    def landfallCheck(self, index):
        indexOfLandfall = index
        try:
            indexOfLandfall = int(indexOfLandfall)
        except ValueError:
            indexOfLandfall = input("Sure? Type 'Y'. ")

        return indexOfLandfall
    
    def nearestPointCheck(self):
        inputRange = range(len(self.timestamp))
        s = [mafs.solveDistanceFromManila(self.latitude[i], self.longitude[i]) for i in inputRange]
        j = min(inputRange, key = lambda i: s[i])
        if s[j] <= 300:
            return j
        print("Did not passed near Manila.")
            
    def getMainAttributes(self, indexToStart):
        self.landfallFlag = False
        
        if type(indexToStart) == int:
            self.landfallFlag = True
            if config.landfall_mode == True and config.nearest_point_mode == False:
                print('{} made Landfall! in: {:.2f}, {:.2f}'.format(self.typhoon_name,
                                    self.latitude[indexToStart], self.longitude[indexToStart]))
                
            # update values
            after = indexToStart + (24/dt)
            before = indexToStart - (24/dt)
            update = lambda attriList: [val for i, val in enumerate(attriList)
                                if i <= after and i >= before and i >= 0]
            rounder = lambda attriList: [round(val, 2) for val in attriList]
            
        # Original params
            self.timestamp = update(self.timestamp)
            if len(self.timestamp) == ( (2*(24/dt)) + 1 ): # Force iff (t-24h, t+24h) is true
                self.latitude = rounder(update(self.latitude))
                self.longitude = rounder(update(self.longitude))
                self.pressure = update(self.pressure)
                self.maxWind = update(self.maxWind)
                
            # Inferred params
                for i in range(len(self.timestamp)):
                    r = mafs.solveDistanceFromManila(self.latitude[i], self.longitude[i])
                    self.distManila = np.append(self.distManila, r)
                    if i == 0:
                        self.displacement = np.append(self.displacement, 0)
                        self.velocity = np.append(self.velocity, 0)
                    if i > 0:
                        s = mafs.solveArcLength(self.latitude[i-1], self.latitude[i],
                                                self.longitude[i-1], self.longitude[i])
                        self.displacement = np.append(self.displacement, s)
                        v = mafs.solveTranslationalVelocity(s)
                        self.velocity = np.append(self.velocity, v)
                        
            # From CSV
                self.precip = csv_ops.extractPrecip(config.csvFile, self.timestamp)
                self.precip = np.asarray(self.precip)
                self.precip = np.multiply(self.precip, dt)
                self.windAt10m = csv_ops.extractWindAt10m(config.csvFile, self.timestamp)
                self.windAt10m = np.asarray(self.windAt10m)
            else:
                print("Not enough data points.")
        else:
            if config.landfall_mode == True and config.nearest_point_mode == False:
                print("No landfall.")
            if config.nearest_point_mode == False and config.landfall_mode == True:
                print("Not near Metro Manila.")
    
    df = []
    def pandaTable(self):
        if self.landfallFlag == True:
            self.df = pd.DataFrame({'ID': self.int_id,
                                    'Name': self.typhoon_name, 
                                    'Timestamp': [self.timestamp],
                                    'Latitude': [self.latitude],
                                    'Longitude': [self.longitude],
                                    'Pressure': [self.pressure],
                                    'Max Wind': [self.maxWind],
                                    'Wind at 10m': [self.windAt10m],
                                    'Displacement': [self.displacement],
                                    'Translational Velocity': [self.velocity],
                                    'Distance From Manila': [self.distManila],
                                    'Rainfall': [self.precip]})
            self.df = self.df.set_index('ID')
            return self.df
        else:
            return pd.DataFrame()
    
    def showTrack(self, createLabel = False):
        self.createMap()
        long, lat = self.mapa(self.longitude, self.latitude)
        self.mapa.plot(long, lat, color='#B90F1B', linewidth=1.5,
                       label = self.typhoon_name)
        self.mapa.scatter(long, lat, marker='o', color='#B90F1B')
        plt.legend()
        
        if createLabel == True:
            for i in range(len(lat)):
                label = i
                plt.annotate(label, (long[i], lat[i]),
                             xytext = (0, 10), textcoords = "offset points",
                             ha = 'center')
        plt.show()
        
    mapa = None
    base_map = None
    def createMap(self):
        print("==================\nInitializing map...")
        self.base_map = plt.figure(figsize=(8, 8))
        self.base_map = config.mapForTrack
        self.base_map.drawmapboundary(fill_color='#87cdf6')
        self.base_map.fillcontinents(color='#20c506',lake_color='#87cdf6')
        self.base_map.drawcoastlines()
        self.mapa = self.base_map
    
    def getDataLines(self, internationalID):
        j = self.internationalID.index('%s'%internationalID)
        self.int_id = self.internationalID[j]
        self.typhoon_name = self.typhoonName[j]
        self.data_count = int(self.dataLine[j])
        self.data_lines = []

        count = 0
        begin_get = 0
        end_get = 0
        line_start = 0
        total_lines = self.countTotalLines() 
        header_lists = self.getHeaderLines()[0]
        
        for i, line in enumerate(header_lists):
            line_ID_check = header_lists[i][6:10]
            if line_ID_check == self.internationalID[j]:                        #Find header line
                line_start = line
        
        with open(self.file) as myFile:                                         #Find line start number
            for line in myFile:
                line_read = line
                while count < (total_lines + 1):
                    if line:
                        line_read.split()
                        line_read = " ".join(line_read.split())
                        if line_read == line_start:                             #Take where to start
                            begin_get = count + 1
                            end_get = count + self.data_count + 1
                        count += 1
                    line_read = myFile.readline()      
                    
        with open(self.file,"r") as myFile:                                     #Extracts the data lines
            for line in myFile.readlines()[begin_get:end_get]:
                line_read = line
                line_read.split()
                line_read = " ".join(line_read.split())
                line_read = list(line_read.split(" "))
                line_read = line_read[:7]
                self.data_lines.append(line_read)
        
        print("==============================")
        print("Data lines for %s(%s) extracted. Total: %i" %(self.typhoon_name,
                                        self.int_id, self.data_count))
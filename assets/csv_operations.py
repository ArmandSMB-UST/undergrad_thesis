# -*- coding: utf-8 -*-
"""
Created on Sun Feb  21 23:48:37 2022

@author: ArmandSMB
"""
import csv
row_data = []

def extractPrecip(filename, timestamp):
    i = 0
    
    global row_data
    row_data = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if i < len(timestamp):
                if row['YYMMDDHH'] == timestamp[i]:   
                    row_data.append(float(row['PRECIP']))
                    i += 1
    return row_data

def extractWindAt10m(filename, timestamp):
    i = 0
    
    global row_data
    row_data = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if i < len(timestamp):
                if row['YYMMDDHH'] == timestamp[i]:   
                    row_data.append(float(row['WS10M']))
                    i += 1
    return row_data

def extractTimestamp(filename, timestamp):
    global row_data
    row_data = []
    start = timestamp[0]
    start_flag = False
    stop = timestamp[len(timestamp)-1]

    with open(filename, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['YYMMDDHH'] == start:
                row_data.append(str(row['YYMMDDHH']))
                start_flag = True
            elif start_flag == True:
                row_data.append(str(row['YYMMDDHH']))
                if row['YYMMDDHH'] == stop:
                    start_flag = False
    return row_data

def resolutionAdjust(row, hourInterval):
    global row_data
    row_data.append(float(row['PRECIP']))
    count = int(row['HH']) 
    if count % hourInterval == 0:                                               #to adjust time interval
        row['YEAR'] = str(row['YEAR'][2:]).zfill(2)
        row['MM'] = str(row['MM']).zfill(2)
        row['DD'] = str(row['DD']).zfill(2)
        row['HH'] = str(row['HH']).zfill(2)
        row['PRECIP'] = sum(row_data)
        row_data = []
        return row
        
def writeTrimmed(newFilename, filename, hourInterval, fieldnames):
    with open(filename, 'r') as csv_file:
        skipHeader(csv_file)
        csv_reader = csv.DictReader(csv_file)     
        with open(newFilename, 'w', newline ='') as new_file:
            csv_writer = csv.DictWriter(new_file, fieldnames)
            csv_writer.writeheader()
            for row in csv_reader:
                rowToWrite = resolutionAdjust(row, hourInterval)
                if rowToWrite != None:
                    rowToWrite['YYMMDDHH'] = (row['YEAR'] +  row['MM'] +
                                                row['DD'] +  row['HH'])
                    del rowToWrite['YEAR'], rowToWrite['MM'], \
                        rowToWrite['DD'], rowToWrite['HH']
                    csv_writer.writerow(rowToWrite)
                        
def skipHeader(file):
    i = 0
    while i < 10:
        i += 1
        next(file)
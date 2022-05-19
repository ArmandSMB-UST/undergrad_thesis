# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:32:15 2022

@author: ArmandSMB
"""
import assets.csv_operations as csv_ops
import assets.config as config

textFile = "raw/bst_2010_to_2020.txt"
lookup = "66666 1001  017 0001 1001 0 6                OMAIS              20100512        "

def findAndRetLine(filePath, lineToFind):                                       #takes starting line to keep
    with open(filePath) as myFile:
        for num, line in enumerate(myFile):
            if lineToFind in line:
                print("Found line!\nLookup at: Line %i" % (num))
                ret_num = num
                break
    return ret_num
            
ret_num = findAndRetLine(textFile, lookup)

with open(textFile, "r") as myFile:
    lines = myFile.readlines()
    pointer = 1
    with open(textFile, "w") as myFileWrite:
        for line in lines:
            if pointer > ret_num:                                               #to trim old data
                myFileWrite.write(line)
            pointer += 1            
    print("All lines after Line %i were saved." % (ret_num) )
    
# =============================================================================
#                               CSV Data Processing 
# =============================================================================

dt = config.dt
csvFile = "raw/Science_Garden_data_2010_to_2021.csv"
trimmedCSVFile = "raw/Science_Garden_data_2010_to_2021_" + str(dt) + "h_resolution.csv"
fieldnames = ['YYMMDDHH', 'PRECIP', 'WS10M']
csv_ops.writeTrimmed(trimmedCSVFile, csvFile, dt, fieldnames)


# =============================================================================
#                           Define Coastlines
# =============================================================================
# This is to determine if landfall happened
path = 'raw'
config.save_coastal_data(path)

# =============================================================================
#                               JSON to CSV
# =============================================================================
import pandas as pd

def jsonToCSV(file):
    write_filename = 0
    with open(file, encoding='utf-8') as inputfile:
        df = pd.read_json(inputfile, orient='index')
    
        write_filename = file[:len(file)-5] 
    df.to_csv(write_filename + '_300.csv', encoding='utf-8', index=True)

nearest_point = 'processed/science-garden-nearest-as-reference400.json'
landfall = 'processed/landfall-as-reference.json'

jsonToCSV(nearest_point)

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:45:39 2022

@author: ArmandSMB
"""
import pandas as pd
import time
import numpy as np
import assets.config as config
from assets.typhoon_classes import header_data, data_lines
    
def main():
    headers = header_data(internationalID = [], dataLine = [], typhoonName = [])
    headers.getHeaders(config.textFile)
    
    datas = data_lines(headers.internationalID, headers.dataLine, headers.typhoonName)
    datas.dataFile(config.textFile)
    
    df_final = pd.DataFrame(columns = config.column_names)

    start_time = time.time()
    n = 0
    for i in range(len(headers.internationalID)):
        if i <= 274:
            datas.getDataLines(headers.internationalID[i])
            datas.intializeAttriPreLandfallCheck()                              # Takes original timestamp and coords
            
            #datas.showTrack(createLabel = True, save = True)
            if config.landfall_mode == True and config.nearest_point_mode == False:
                index = input('Enter index of landfall: ')
                index = datas.landfallCheck(index)
                
            elif config.nearest_point_mode == True and config.landfall_mode == False:
                index = datas.nearestPointCheck()
            
            datas.getMainAttributes(index)
            
            if len(datas.timestamp) == 2 * (24/config.dt) + 1 and datas.landfallFlag == True:
                df, n = datas.pandaTable(n) # n is for indexing
                if not df.empty:
                    df_final = pd.concat([df_final, df], axis=0)
    print("Execution time: %.3f s \n" % (time.time() - start_time)) 
    print('=====================================================')

    if config.landfall_mode == True and config.nearest_point_mode == False:
        df_final.to_json('processed/landfall-as-reference.json', orient = 'index', indent = 2)
    elif config.nearest_point_mode == True and config.landfall_mode == False:
        df_final.to_json('processed/science-garden-nearest-as-reference400.json', orient = 'index', indent = 2)
    return df_final
    
def trackSee(dataframe, ID, showLegend):
    import assets.track_visualize as track_visual
    track_visual.trackVisualize(dataframe, ID, showLegend)    

if __name__ == '__main__':
    df_FINAL = main()
# =============================================================================
#     headers = header_data(internationalID = [], dataLine = [], 
#                           typhoonName = [])
#     headers.getHeaders(config.textFile)
#     
#     datas = data_lines(headers.internationalID, headers.dataLine, headers.typhoonName)
#     datas.dataFile(config.textFile)
#     
#     df_final = pd.DataFrame(columns = config.column_names)
# 
#     start_time = time.time()
#     n = 0
#     datas.getDataLines(2002)
#     datas.intializeAttriPreLandfallCheck()                              # Takes original timestamp and coords
#     
#     
#     import matplotlib.pyplot as plt
#     # This is how you change the font in matplotlib plots
#     font = {'family' : 'Arial',
#             'weight' : 'normal',
#             'size'   : 9}
#     plt.rcParams["axes.labelweight"] = "bold"
#     plt.rc('font', **font)
#     datas.showTrack(createLabel=True)
#             
#     #datas.showTrack(createLabel = True, save = True)
#     if config.landfall_mode == True and config.nearest_point_mode == False:
#         index = input('Enter index of landfall: ')
#         index = datas.landfallCheck(index)
#         
#     elif config.nearest_point_mode == True and config.landfall_mode == False:
#         index = datas.nearestPointCheck()
#         print(index)
#     
#     datas.getMainAttributes(index)
#     #datas.showTrack(createLabel=True)
# =============================================================================

    pass
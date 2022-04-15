# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:45:39 2022

@author: ArmandSMB
"""
import pandas as pd
import time
import assets.config as config
from assets.typhoon_classes import header_data, data_lines
    
if __name__ == '__main__':
    headers = header_data(internationalID = [], dataLine = [], 
                          typhoonName = [])
    headers.getHeaders(config.textFile)
    
    datas = data_lines(headers.internationalID, headers.dataLine, headers.typhoonName)
    datas.dataFile(config.textFile)
    
    column_names = config.column_names
    df_final = pd.DataFrame(columns=column_names)
    
    

    start_time = time.time()
    for i in range(len(headers.internationalID)):
        if i <= 274:
            datas.getDataLines(headers.internationalID[i])
            datas.intializeAttriPreLandfallCheck()                              # Takes original timestamp and coords
            
            datas.showTrack(createLabel = True)
            
            if config.landfall_mode == True and config.nearest_point_mode == False:
                index = input('Enter index of landfall: ')
                index = datas.landfallCheck(index)
                
            elif config.nearest_point_mode == True and config.landfall_mode == False:
                index = datas.nearestPointCheck()
            
            datas.getMainAttributes(index)
            
            df = datas.pandaTable()
            if not df.empty:
                df_final = pd.concat([df_final, df], axis=0)
    print("Execution time: %.3f s \n" % (time.time() - start_time)) 
    print('=====================================================')
            
    if config.landfall_mode == True and config.nearest_point_mode == False:
        df_final.to_json('processed/landfall-as-reference.json', orient = 'index', indent = 2)
    elif config.nearest_point_mode == True and config.landfall_mode == False:
        df_final.to_json('processed/nearest-as-reference.json', orient = 'index', indent = 2)
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:45:39 2022

@author: ArmandSMB
"""
import time
import assets.config as config
from assets.typhoon_classes import header_data, data_lines
import assets.intense_maths as mafs
    
if __name__ == '__main__':
    start_time = time.time()
    filePath = config.textFile
    headers = header_data(internationalID = [], dataLine = [], 
                          typhoonName = [])
    headers.getHeaders(filePath)
    
    datas = data_lines(headers.internationalID, headers.dataLine, 
                       headers.typhoonName)
    datas.dataFile(filePath)
    
    datas.getDataLines(1104)
    datas.getMainAttributes()
    #datas.showTrack()
    
    df = datas.pandaTable()
    
    print("Execution time: %.3f s" % (time.time() - start_time)) 

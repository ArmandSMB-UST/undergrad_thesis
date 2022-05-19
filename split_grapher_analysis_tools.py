# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:21:09 2022

@author: ArmandSMB
"""
import scipy.stats as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This is how you change the font in matplotlib plots
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 9}
plt.rcParams["axes.labelweight"] = "bold"
plt.rc('font', **font)

trackBaseClr = ['#CC0005', '#00A29C', '#CE7700', '#B9009C', '#35B100','b']
trackHighClr = ['r', 'g', 'b', 'm', 'orange', '#e2e923', 'b',  '#F400CE']
markers = ['o','v','^','s','D','p']
style = ['-','--','-','--','-','--']
     
letters = ["(a)", "(b)" ,"(c)", "(d)", "(e)", "(f)"]
baseColors = ['#CC0005', '#FD1F25']
hlghtColors = ['#00C7BF', '#00F0E7']

def in_outExtract(dataframe):
    inputs = dataframe.iloc[:, 1:df.shape[1]-1]
    rainfall = dataframe.iloc[:, [-1]]
    return inputs, rainfall

def generateRPTable(dataframe):
    ins, rainfall = in_outExtract(dataframe)
    
    linregress_df = pd.DataFrame(columns = ['parameter with rainfall', 'slope', 'intercept',
                                            'R', 'R2', 'p', 'std error'])
    for i in range(len(ins.columns)):
        res = sp.linregress(ins.iloc[:,i].values, rainfall.T)
        temp_df = pd.DataFrame([{'parameter with rainfall': ins.columns[i], 'slope': res.slope,
                                'intercept': res.intercept, 'R': res.rvalue,
                                'R2': res.rvalue**2, 'p': res.pvalue,
                                'std error': res.stderr}])
        linregress_df = pd.concat([linregress_df, temp_df], ignore_index = True)
    return linregress_df

def scatterMatrix(dataframe):
    plt.rcParams["figure.dpi"] = 300
    pd.plotting.scatter_matrix(dataframe.iloc[:, 1:], alpha = 0.2,
            marker = '.', diagonal = 'hist', figsize = (8,7),
            range_padding = 0.22)
    
def concatMaster(input_df, data_set_ids, df_to_get_data):
    reference_ids = df['ID'].values.tolist()
    for i in range(len(data_set_ids)):
      if data_set_ids[i] in reference_ids:
        start_index = reference_ids.index(data_set_ids[i])
        j = start_index
        while j <= start_index + 16:
          input_df = pd.concat([input_df, df_to_get_data.iloc[[j],:]], axis = 0, ignore_index = True)
          j+= 1
    return input_df

def datasetSplitter(df, train_size = 0.875):
    ids = np.unique(df['ID'])
    print("From {} amount of TCs. Splitting data...\n".format(len(ids))) 
    print("train: {}, test: {}".format(len(ids)*train_size, len(ids)*(1-train_size)))
    if len(ids) * train_size % 1 == 0:
        inputs = df.iloc[:,:9] # Change to 8 if no longitude
        rf = df.iloc[:, [-1]]
        
        from sklearn.model_selection import train_test_split
    
        ids_train, ids_test, temp_train, temp_test = train_test_split(ids, 
                  np.arange(0, len(ids), 1), train_size = train_size, random_state = 12)
      
        input_train = pd.DataFrame(columns = [col for col in inputs.columns])
        output_train = pd.DataFrame(columns = [col for col in rf.columns])
        
        input_train = concatMaster(input_train, ids_train, df_to_get_data = inputs)
        output_train = concatMaster(output_train, ids_train, df_to_get_data = rf)
        
        input_test = pd.DataFrame(columns = [col for col in inputs.columns])
        output_test = pd.DataFrame(columns = [col for col in rf.columns])
        
        input_test = concatMaster(input_test, ids_test, df_to_get_data = inputs)
        output_test = concatMaster(output_test, ids_test, df_to_get_data = rf)
        
        return input_train, input_test, output_train, output_test
    elif len(ids) * train_size % 1 != 0:
        print("Modify train_size such that train and test are whole numbers.")

def trainTestSaver(input_train, input_test, output_train, output_test, train_size):
    varName = lambda p: [k for k, v in globals().items() if id(p) == id(v)]
    input_train.to_csv('processed/{}_{}.csv'.format(varName(input_train), train_size))
    output_train.to_csv('processed/{}_{}.csv'.format(varName(output_train), train_size))
    input_test.to_csv('processed/{}_{}.csv'.format(varName(input_test), train_size))
    output_test.to_csv('processed/{}_{}.csv'.format(varName(output_test), train_size))
    print("Saved csv versions of train and test set.")
  
def visualizeTotalRainfall(data, ID = None, graph = True):
    import assets.config as config
    timeseries = np.arange(-24, 27, config.dt)
    rainfall_total = [val for j,val in enumerate(data['rf'])
                      if ID == data['ID'][j]]
    total = sum(val for val in rainfall_total)
    idRain_df = pd.DataFrame({'ID': ID, 'rf': total}, index = [0])
    
    if graph != False:
        plt.xlim(-24, 24)
        plt.text(-21, max(rainfall_total)*0.30,
             "Total Rainfall: {:.4f}mm".format(total), fontsize = 12)
        plt.title("Rf for ID:{}".format(ID), fontsize = 16)
        plt.bar(timeseries[1:], rainfall_total[1:], tick_label = timeseries[1:],
                   width = -3.0, align = 'edge', edgecolor = '#000000', alpha = 1)
        plt.show()
    return idRain_df

def histogrammer(data, ID, data2 = pd.DataFrame(data=None)):
    import assets.config as config
    timeseries = np.arange(-24, 27, config.dt)
    
    rainfall_total = [val for j, val in enumerate(data['rf'])
                if ID == data['ID'][j]]
    rainfall_pred_total = [val for j, val in enumerate(data['rf_pred'])
                if ID == data['ID'][j]]
    if data2.empty == False:
        rainfall_pred_ext = [val for j, val in enumerate(data2['rf_pred'])
                if ID == data2['ID'][j]]
    
    print("==================")
    print("ID {}".format(ID))
    actual = sum(val for val in rainfall_total)
    predicted = sum(val for val in rainfall_pred_total)

    total_error = (actual - predicted) * 100 / actual
    print("Total Rainfall: {} mm".format(actual))
    print("Total Predicted Rainfall: {} mm".format(predicted))
    
    plt.ylim(min([min(rainfall_pred_total), min(rainfall_total)]),
             max([max(rainfall_pred_total), max(rainfall_total)]))
    plt.xlim(-24, 24)
# =============================================================================
#     plt.text(-21, ax.get_ylim()[1] + 0.075*(ax.get_ylim()[0] - ax.get_ylim()[1]),
#          "Total Error: {:.4f}%".format(total_error))
# =============================================================================
    plt.title("Rf for Pred (b) v. Actual (o) ID:{}".format(ID))
    if data2.empty == False:
        plt.bar(timeseries[1:], rainfall_pred_ext[1:], tick_label = timeseries[1:],
           width = -3.0, align = 'edge', alpha = 1,
           edgecolor = '#000000', color = baseColors[0]) 
    plt.bar(timeseries[1:], rainfall_pred_total[1:], tick_label = timeseries[1:],
           width = -3.0, align = 'edge', alpha = 0.5,
           edgecolor = '#000000', color = baseColors[1]) 
    plt.bar(timeseries[1:], rainfall_total[1:], tick_label = timeseries[1:],
           width = -3.0, align = 'edge', alpha = 0.35,
           edgecolor = '#000000', color = hlghtColors[0])    
        
def scatterXY(data, ID):
    # Plots the actual as X then Y as prediction
    pred = [val for j, val in enumerate(data['rf_pred']) if ID == data['ID'][j]]
    obs = [val for j, val in enumerate(data['rf']) if ID == data['ID'][j]]
    res = sp.linregress(obs, pred)
    plt.scatter(obs, pred, color = '#00F0E7')
    line_x = np.linspace(0, max(obs), 2)
    name = database['Name'][database['ID']==ID]
    plt.title("{} - {}".format(ID,name.iloc[1]))
    plt.plot(line_x, res.intercept + res.slope*line_x, linestyle = '--',
             color = '#00A29C')
    print(res.rvalue**2)
        
def seeRainfall(dataFrame):
    ids = np.unique(dataFrame['ID'])
    id_rf_df = pd.DataFrame(columns = ['ID', 'Rainfall'])
    
    for i in range(len(ids)):
        temp_df = visualizeTotalRainfall(dataFrame, ID = ids[i], graph = False)
        id_rf_df = pd.concat([id_rf_df, temp_df], axis = 0, ignore_index=True)
        
    plt.scatter([i for i in range(len(id_rf_df['Rainfall']))], id_rf_df['Rainfall'])
    plt.show()
    
def RMSE(obs, pred):
    import math
    n = len(obs)
    rmse = sum((obs[i]-pred[i])**2 for i in range(n))
    return math.sqrt(rmse/n)
    
def timePlot(data, stepData, ID, title):
    time = [val for j, val in enumerate(data['t']) if ID == data['ID'][j]]
    pred1 = [val for j, val in enumerate(data['rf_pred']) if ID == data['ID'][j]]
    pred2 = [val for j, val in enumerate(stepData['rf_pred']) if ID == stepData['ID'][j]]
    obs = [val for j, val in enumerate(data['rf']) if ID == data['ID'][j]]


    print("==================")
    print("ID {}".format(ID))
    actual = sum(val for val in obs)
    predicted = sum(val for val in pred1)
    predicted2 = sum(val for val in pred2)
    total_error1 = abs((actual - predicted)) * 100 / actual
    total_error2 = abs((actual - predicted2)) * 100 / actual
    print("Line RMSE: {}".format(RMSE(obs,pred1)))
    print("=====")
    print("Step RMSE: {}".format(RMSE(obs,pred2)))
    print("Total Rainfall: {} mm".format(actual))
    print("Total Predicted Linear: {} mm".format(predicted))
    print(total_error1)
    print("Total Predicted Step: {} mm".format(predicted2))
    print(total_error2)
    
    plt.plot(time, obs, linestyle = '--', marker = '.',
             color = hlghtColors[0])
    plt.plot(time, pred1, color = baseColors[1])
    plt.plot(time, pred2, color = 'm')
    
    plt.xlabel('Time (h)')
    plt.ylabel('Rainfall (mm)')
    name = database['Name'][database['ID']==ID]
    plt.title("{} - {}".format(ID, name.iloc[1]), pad=3.0)
    plt.text(-41, ax.get_ylim()[1] -
             0.02 *(ax.get_ylim()[0] - ax.get_ylim()[1]),
             title, weight="bold", fontsize = 12) # use text instead of title; it is easier to use

def scattered(node_x, node_y, color = 'b', marker = 'o', alpha = 1):
    plt.scatter(node_x, node_y, color = color, marker = marker, s = 25,
                alpha = alpha)

def bestfit(node_x, node_y, color = 'b'):
    x = np.array([0, max(node_x)])
    node_best = np.polyfit(node_x, node_y, 1)
    plt.plot(x, node_best[0] * x + node_best[1], linestyle = '--', color = color,
             linewidth = 1)

def PLOT(data1, i, color = baseColors[1], data2 = pd.DataFrame(data=None)):
    node_x = [val for j, val in enumerate(data1['rf']) if ids[i] == data1['ID'][j]]
    node_y = [val for j, val in enumerate(data1['rf_pred']) if ids[i] == data1['ID'][j]]

    if data2.empty == False:
        y2 = [val for j, val in enumerate(data2['rf_pred']) if ids[i] == data2['ID'][j]]
        scattered(node_x, y2, color = '#BF00BF', alpha = 0.8)
        res = sp.linregress(node_x, y2)
        bestfit(node_x, y2, color ='#BF00BF')
        
    scattered(node_x, node_y, color = color, alpha = 0.8)
    result = sp.linregress(node_x, node_y)
    bestfit(node_x, node_y , color = color)
    #print("ID {}, {}".format(ids[i], result.rvalue**2))
    name = database['Name'][database['ID'] == ids[i]]
    plt.title("{}-{}".format(ids[i], name.iloc[1]), pad=3.0)
    
    if data2.empty == False: 
        plt.text(ax.get_xlim()[1] - 0.38*(ax.get_xlim()[1]-
                                             ax.get_xlim()[0]), 
                 ax.get_ylim()[0] - 0.05*(ax.get_ylim()[0] - 
                                          ax.get_ylim()[1]),
             r"$R^2 = {:.3f}$".format(res.rvalue**2), fontsize = 9,
             color = '#960096', weight = 'bold')
# =============================================================================
#     plt.text(ax.get_xlim()[1] - 0.38*(ax.get_xlim()[1]-
#                                              ax.get_xlim()[0]), 
#              ax.get_ylim()[0] - 0.15*(ax.get_ylim()[0] - 
#                                       ax.get_ylim()[1]),
#              r"$R^2 = {:.3f}$".format(result.rvalue**2), fontsize = 9,
#              color = '#B80005', weight = 'bold')
# =============================================================================
    plt.text(ax.get_xlim()[1] - 0.45*(ax.get_xlim()[1]-
                                             ax.get_xlim()[0]), 
             ax.get_ylim()[0] - 0.05*(ax.get_ylim()[0] - 
                                      ax.get_ylim()[1]),
             r"$R^2 = {:.3f}$".format(result.rvalue**2), fontsize = 9,
             color = '#000000', weight = 'bold')
    # Letter Title
    plt.text(-0.3*ax.get_xlim()[1], ax.get_ylim()[1] - 0.08*(ax.get_ylim()[0] - 
        ax.get_ylim()[1]), letters[i], weight="bold",
             fontsize = 12)
    plt.xlabel('Actual (mm)')
    plt.ylabel('Predicted (mm)')
    return result

def trackVisualize(dataframe, id_in,k, showLegend = False):
    import assets.config as config
    import assets.intense_maths as mafs

    base_map = config.mapForTrack3
    base_map.drawmapboundary(fill_color='#87cdf6')
    base_map.fillcontinents(color='#4ce053',lake_color='#87cdf6')
    base_map.drawcoastlines(color = '#383838')
    base_map.drawmeridians(np.linspace(113.5,128.5, 4), labels=[0,0,0,1],
                           color = '#757575')
    base_map.drawparallels(np.linspace(11, 21, 5), labels = [1,0,0,0],
                           color = '#757575')
    base_map.scatter(121, 14.7, color = 'r', latlon = True,
                     marker = 'o', s=100, alpha = 0.6, label = "Metro Manila (POI)")
    for i in range(len(id_in)):
        long = [val for j, val in enumerate(dataframe['Longitude']) if id_in[i] == dataframe['ID'][j]]
        lat = [mafs.corPamToLat(val) for j, val in enumerate(dataframe['Coriolis Parameter'])
               if id_in[i] == dataframe['ID'][j]]
        long, lat = base_map(long, lat)
        
        for m in range(len(lat)):
                label = m
                if id_in[i] == 1501:
                    plt.annotate(label, (long[m], lat[m]),
                                 xytext = (5, 10), textcoords = "offset points",
                                 ha = 'left', color = 'white')
                elif id_in[i] == 1306:
                    plt.annotate(label, (long[m], lat[m]),
                                 xytext = (-5, -10), textcoords = "offset points",
                                 ha = 'right', color = 'white')
        
        name = dataframe['Name'][dataframe['ID']==id_in[i]]
        base_map.plot(long, lat, color = trackHighClr[i+k], linestyle = style[i+k],
              linewidth=1, label = "{} ({})".format(name.iloc[1],id_in[i]),
              marker = markers[i+k], ms=5,path_effects=[pe.Stroke(linewidth=2, foreground='black'), pe.Normal()])
    
    #,path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()]
# =============================================================================
#         base_map.scatter(long, lat, marker=markers[i], 
#                          color = trackHighClr[i],s=6)
# =============================================================================
    if showLegend == True:
        plt.legend(loc='lower left')
        
def plotExpRes(dataFrame):
    res_var = [val for val in dataFrame['rf']]
    letters = ["(a)", "(b)" ,"(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for j, col in enumerate(dataFrame.columns):        
        if col != "ID" and j != 9:
            exp_var = [val for val in dataFrame[col]]
            result = sp.linregress(exp_var, res_var)

            ax = plt.subplot(2,4,int(j))
            plt.scatter(exp_var, res_var, alpha = 0.2, s=10)
            
            x = np.array([min(exp_var), max(exp_var)])
            plt.plot(x, result.slope * x + result.intercept,
                     linestyle = '--', color = "r", linewidth = 2)
            plt.text(ax.get_xlim()[1] - 0.90*(ax.get_xlim()[1]-
                                             ax.get_xlim()[0]),
                     ax.get_ylim()[0] - 0.85*(ax.get_ylim()[0] - 
                                              ax.get_ylim()[1]),
                     r"$R = {:.3f}$".format(result.rvalue), 
                     fontsize = 9)
# =============================================================================
#             plt.text(ax.get_xlim()[1] - 0.90*(ax.get_xlim()[1]-
#                                              ax.get_xlim()[0]),
#                      ax.get_ylim()[0] - 0.75*(ax.get_ylim()[0] - 
#                                               ax.get_ylim()[1]),
#                      r"$R^2 = {:.3f}$".format(result.rvalue**2), 
#                      fontsize = 9)
# =============================================================================
            # Letter Title
            plt.text(ax.get_xlim()[1] - 1.25*(ax.get_xlim()[1]-
                                             ax.get_xlim()[0]),
                     ax.get_ylim()[1] - 0.01*(ax.get_ylim()[0] - 
                                              ax.get_ylim()[1]), 
                     letters[j-1], weight="bold",
                     fontsize = 14)
            
            plt.ylim(min(res_var), max(res_var))
            plt.xlim(min(exp_var), max(exp_var))
            plt.xlabel(col)
            plt.ylabel("rf")
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()
    
def boundaryConditioner(dataFrame, boundaryRadius, saveCSV = False):
    dataFrame.loc[dataFrame['r'] >= boundaryRadius, 'rf'] = 0
    if saveCSV == True:
        varName = lambda p: [k for k, v in globals().items() if id(p) == id(v)]
        dataFrame.to_csv("processed/final2/{}.csv".format(varName(dataFrame)))
        
def rSegregate(dataFrame, interval, limit):
    r_vals = np.linspace(0, limit, int(limit/interval + 1))
    varName = lambda p: [k for k, v in globals().items() if id(p) == id(v)]
    for i in range(len(r_vals)):
        if r_vals[i] != 0:
            temp_df = pd.DataFrame(columns = dataFrame.columns)
            if i != max(range(len(r_vals))):
                temp_df = pd.concat([temp_df, dataFrame.loc[dataFrame['r'] <= r_vals[i]]])
                temp_df = temp_df.loc[temp_df['r'] > r_vals[i-1]]
                temp_df.to_csv("processed/final/{}_r{}-{}.csv".format(varName(dataFrame),
                                               r_vals[i-1], r_vals[i]))
            elif i == max(range(len(r_vals))):
                temp_df = pd.concat([temp_df, dataFrame.loc[dataFrame['r'] <= r_vals[i]]])
                temp_df = temp_df.loc[temp_df['r'] > r_vals[i-1]]
                temp_df.to_csv("processed/final/{}_r{}-{}.csv".format(varName(dataFrame),
                                               r_vals[i-1], r_vals[i]))
                temp_df = pd.DataFrame(columns = dataFrame.columns)
                temp_df = pd.concat([temp_df, dataFrame.loc[dataFrame['r']>= r_vals[i]]])
                temp_df.to_csv("processed/final/{}_r{}.csv".format(varName(dataFrame),
                                               r_vals[i]))

def errorPlot(DF, i):
    obs = [val for j, val in enumerate(DF['rf']) if ids[i] == DF['ID'][j]]
    pred = [val for j, val in enumerate(DF['rf_pred']) if ids[i] == DF['ID'][j]]
    errs = [obs[j]-pred[j] for j in range(len(obs))]
    
    letters = ["(a)", "(b)" ,"(c)", "(d)", "(e)", "(f)"]
    name = database['Name'][database['ID'] == ids[i]]
    
    plt.title("{}-{}".format(ids[i], name.iloc[1]), pad=3.0)
    
    plt.plot([-27, 27], [0,0], color = '#757575', linestyle='--', alpha = 0.8)
    plt.plot(np.linspace(-24, 27, 17), errs, marker = '.', label = ids[i])
    plt.xlim(-24, 24)
    
    plt.text(ax.get_xlim()[1] - 1.25*(ax.get_xlim()[1]-
                                ax.get_xlim()[0]),
        ax.get_ylim()[1] + 0.05*(ax.get_ylim()[1] - 
                                 ax.get_ylim()[0]), 
        letters[i], weight="bold", fontsize = 11)
    plt.xlabel("Time (h)")
    plt.ylabel("Error $rf$ (mm)")

if __name__ == '__main__':
    database = pd.read_csv("processed/final2/['database'].csv", index_col=[0])
    df = database.drop(['Name', 'Timestamp', 'Latitude'], axis = 1)
    mod_data = pd.read_csv("processed/final2/test_results_lin.csv", index_col = [0])
    step_data = pd.read_csv("processed/final2/test_results_step_lin.csv", index_col = [0])
    
# =============================================================================
#     fig = plt.figure(figsize=(6,4), dpi = 300)
#     fig.tight_layout(h_pad = 10)
#     ids = np.unique(mod_data['ID'])
#     
#     ax = plt.subplot(231, adjustable = 'box')
#     errorPlot(mod_data, 0)
#     
#     ax = plt.subplot(232, adjustable = 'box')
#     errorPlot(mod_data, 1)
#     
#     ax = plt.subplot(233, adjustable = 'box')
#     errorPlot(mod_data, 2)
#     
#     ax = plt.subplot(234, adjustable = 'box')
#     errorPlot(mod_data, 3)
#     
#     ax = plt.subplot(235, adjustable = 'box')
#     errorPlot(mod_data, 4)
#     
#     ax = plt.subplot(236, adjustable = 'box')
#     errorPlot(mod_data, 5)
#     
#     plt.subplots_adjust(hspace=0.5, wspace=0.5)
#     plt.show()
# =============================================================================
# =============================================================================
#     fig = plt.figure(figsize=(9, 4.5), dpi=300)
#     fig.tight_layout(h_pad = 10)
#     plt.subplots_adjust(hspace=0.4, wspace=0.5)
#     plotExpRes(df)
# =============================================================================
    #mod_data = pd.read_csv('processed/likely_final/results/'+
    #                          'test_final_Pred_lin.csv', index_col = [0])
    #step_data = pd.read_csv('processed/final/test_result_r500_then_0.csv', index_col =[0])
# =============================================================================
#     ids = np.unique(mod_data['ID'])
#     
#     fig = plt.figure(figsize=(8.0, 5.), dpi=300)
#     fig.tight_layout(h_pad = 10)
#     
#     ax = plt.subplot(231, adjustable = 'box')
#     #PLOT(mod_data, 0)
#     timePlot(mod_data, step_data, ID = ids[0], title = letters[0])
#     #histogrammer(mod_data, ids[0], data2 = step_data)
#     
#     ax = plt.subplot(232, adjustable = 'box')
#     #PLOT(mod_data, 1)
#     timePlot(mod_data, step_data, ID = ids[1], title = letters[1])
#     #histogrammer(mod_data, ids[1], data2 = step_data)
#     
#     ax = plt.subplot(233, adjustable = 'box')
#     #PLOT(mod_data, 2)
#     timePlot(mod_data, step_data,  ID = ids[2], title = letters[2])
#     #histogrammer(mod_data, ids[2], data2 = step_data)
#     
#     ax = plt.subplot(234, adjustable = 'box')
#     #PLOT(mod_data, 3)
#     timePlot(mod_data, step_data, ID = ids[3], title = letters[3])
#     #histogrammer(mod_data, ids[3], data2 = step_data)
#     
#     ax = plt.subplot(235, adjustable = 'box')
#     #PLOT(mod_data, 4)
#     timePlot(mod_data, step_data, ID = ids[4], title = letters[4])
#     #histogrammer(mod_data, ids[4], data2 = step_data)
#     
#     ax = plt.subplot(236, adjustable = 'box')
#     #PLOT(mod_data, 5)
#     timePlot(mod_data, step_data, ID = ids[5], title = letters[5])
#     #histogrammer(mod_data, ids[5], data2 = step_data)
#     
#     plt.subplots_adjust(hspace=0.5, wspace=0.5) # this adjusts the distance between subplots
#     #plt.savefig("../../LaTeX_writeup/fig/rainfall_pred_line.png")
#     #plt.savefig("../../LaTeX_writeup/fig/pred_v_obs_line.png")
#     plt.show()
# =============================================================================
    
# =============================================================================
#     input_data = pd.read_csv("processed/final2/['database'].csv", index_col = [0])
#     input_data = input_data.drop(['ID','Name','Timestamp','Latitude'], axis=1)
#     scatterMatrix(input_data)
#     #plt.savefig("../../LaTeX_writeup/fig/scatter_matrix.png")
#     plt.show()
# =============================================================================
    #rp_df = generateRPTable(input_data)
    #figure = plt.figure(figsize=(12,5), dpi=300)
    #figure.tight_layout(h_pad = 10)
    #train_set = pd.read_csv("processed/final/nearest300/train.csv", index_col = [0])
    #r_squared = [0.7549,0.4339,0.6199,0.5872,0.5953, 0.4233] #Quadratic models Ma'am Rea
    
    #test_set = pd.read_csv("processed/final/test_final_boundaryCondition.csv", index_col = [0])
    #rSegregate(test_set, 100, 500)
    #boundaryConditioner(test_set, 500, saveCSV = True)
    #plotExpRes(train_set)
# =============================================================================
#     ids = np.unique(train_set['ID'])
#     df = pd.DataFrame(columns = ['ID', 'rf'])
#     for i in ids:
#         DF = visualizeTotalRainfall(train_set, i, graph=False)
#         df = pd.concat([df, DF], axis = 0)
#     figure = plt.figure(figsize=(7, 5), dpi=300)
#     plt.hist(df['rf'], bins = np.linspace(0, 600,25), edgecolor = 'black')
#     plt.ylim(0, 12)
#     plt.xlim(0, 600)
#    # plt.boxplot(df['rf'])
#     plt.xlabel('$rf$ (mm)')
#     plt.title("Training Set Total Rainfall Histogram and Box Plot")
#     plt.show()
# =============================================================================
# =============================================================================
#     ids = np.unique(train_set['ID'])
#     for id_val in ids:
#         visualizeTotalRainfall(train_set, id_val)
# =============================================================================
# =============================================================================
#     test_set = pd.read_csv("processed/likely_final/test_final.csv", index_col = [0])
#     print("Testing set TCs: {}".format(6))
#     print(test_set.head())
# =============================================================================
    
    #graphTracks()
    #mod_data = pd.read_csv('processed/general_model_results_likely_final.csv', index_col = [0])
    #histogrammer(mod_data)    
    
    #total_rain = sum(val for j, val in enumerate(mod_data['rf']) if mod_data['ID'][j] == 1111)
    
# =============================================================================
#     #figure = plt.figure(figsize=(7,5), dpi=300)
#     import matplotlib.patheffects as pe
#     #trackVisualize(database,ids,True)
#     #plt.savefig("../../LaTeX_writeup/fig/test_set_track.png")
#     
#     import matplotlib.patheffects as pe
#     figure = plt.figure(figsize=(7,4), dpi=300)
#     trackVisualize(database, [1306, 1501], 2, True)
#     plt.show()
# =============================================================================
# =============================================================================
#     df = pd.read_csv("processed/final3_400bound/['df'].csv", index_col = [0])
#     trainX, testX, trainY, testY = datasetSplitter(df, train_size = 50/53)
#     trainTestSaver(trainX, testX, trainY, testY, train_size = 50/53)
# =============================================================================
    pass
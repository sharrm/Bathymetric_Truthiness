# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:25:57 2023

@author: sharrm
"""

import datetime
# from keras.metrics import MeanIoU
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd
import pickle
# from torch import tensor
# from torchmetrics import JaccardIndex
# from torchmetrics.classification import MulticlassJaccardIndex
import rasterio
from pytictoc import TicToc
from scipy.ndimage import median_filter #, gaussian_filter
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier # AdaBoostClassifier, 
# from sklearn.metrics import  f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, jaccard_score
from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sys
import time
from yellowbrick.classifier import ROCAUC
# from sklearn.metrics import PrecisionRecallDisplay

t = TicToc()
t.tic()

np.random.seed(42)
rcParams['figure.dpi'] = 600

# %% - globals

current_time = datetime.datetime.now()
log_file = r"P:\Thesis\Test Data\_Logs\Log_" + current_time.strftime('%Y%m%d_%H%M') + '.txt'
# scale = MinMaxScaler()

#-- When information is unavailable for a cell location, the location will be assigned as NoData. 
upper_limit = np.finfo(np.float32).max/10
lower_limit = np.finfo(np.float32).min/10

feature_list = ["w492nm",               #1
                "w560nm",               #2   
                "w665nm",               #3  
                "w833nm",               #4
                "pSDBg",                #5
                "pSDBg_roughness",      #6  
                "pSDBg_stdev_slope",    #7
                "pSDBr"]                #8

# feature_list = ["w492nm",               #1
#                 "w560nm",               #2   
#                 "w665nm",               #3  
#                 "w833nm"]

# feature_list = ["w492nm",               #1
#                 "w560nm",               #2   
#                 #"w665nm",               #3  
#                 "w833nm",               #4
#                 "pSDBg",                #5
#                 "pSDBg_roughness",      #6  
#                 "pSDBg_stdev_slope"]    #7

# feature_list = ['pSDBg',
#                 'pSDBg_roughness',
#                 'pSDBg_stdevslope',
#                 'pSDBr',
#                 'pSDBr_roughness',
#                 'pSDBr_stdevslope'
#                 ]

# labels = {0: 'No Data', 1: 'False', 2:'True'}
tf_labels = {0: 'No Data', 1: 'False', 2: 'True'}
iou_labels = {0: 'No Data', 2: 'True Negative', 3: 'False Negative', 4: 'True Positive', 5: 'False Positive'}

tf_cmap = {0:[225/255, 245/255, 255/255, 1],
           1:[225/255, 175/255, 0/255, 1],
           2:[75/255, 130/255, 0/255, 1]}

iou_cmap = {0:[0/255, 0/255, 0/255, 1],
            2:[225/255, 175/255, 0/255, 1],
            3:[75/255, 180/255, 210/255, 1],
            4:[75/255, 130/255, 0/255, 1],
            5:[170/255, 50/255, 90/255, 1]}


# train - test ------------

# analysis
# Perform_IOU = False # intersection over union
Perform_IOU = True # intersection over union

n_jobs = 6


# %% prediction ------------
    
test_rasters = [r"P:\Thesis\Test Data\TinianSaipan\_8Band\_Composite\Saipan_Extents_NoIsland_composite.tif",
                 r"P:\Thesis\Test Data\Puerto Real\_8Band\_Composite\Puerto_Real_Smaller_composite.tif",
                 r'P:\Thesis\Test Data\GreatLakes\_8Band_Focused\_Composite\GreatLakes_Mask_NoLand_composite.tif',
                 r'P:\Thesis\Test Data\Niihua\_8Band\_Composite\Niihua_Mask_composite.tif']

test_models = ["P:\Thesis\Models\Random Forest (400 Trees)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (800 Trees)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (100 Trees 10 Samples)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (100 Trees 100 Samples)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (100 Trees 1000 Samples)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (100 Trees 10000 Samples)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (10 Trees)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (50 Trees)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (100 Trees)_8Band_2000Trees_4Masks_20230314_1409.pkl",
              "P:\Thesis\Models\Random Forest (200 Trees)_8Band_2000Trees_4Masks_20230314_1409.pkl"]

# IOU metrics
test_masks = [r'P:\Thesis\Masks\Saipan_Mask_NoIsland_TF.tif',
             r"P:\Thesis\Masks\Niihua_Mask_TF.tif",
             r"P:\Thesis\Masks\PuertoReal_Mask_TF.tif",
             r"P:\Thesis\Masks\GreatLakes_Mask_NoLand_TF.tif"]

# Post-Processing
# median_filter_tf = True
median_filter_tf = False

# write_prediction = True
write_prediction = False

# if write_prediction:
#     prediction_path = os.path.abspath(os.path.join(os.path.dirname(predict_raster), '..', '_Prediction'))
    
#     if not os.path.exists(prediction_path):
#         os.makedirs(prediction_path)
    
#     prediction_path = prediction_path + '\\' + os.path.basename(predict_raster).replace('composite.tif', 'prediction_') + current_time.strftime('%Y%m%d_%H%M') + '.tif'
#     prediction_difference_path = prediction_path.replace('prediction_', 'prediction_diff_')

# %% - functions

def tf_colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [0/255, 0/255, 0/255, 1] # unclassified 
    rgba[data==1, :] = [225/255, 175/255, 0/255, 1]
    rgba[data==2, :] = [75/255, 130/255, 0/255, 1]
    return rgba

def iou_colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [0/255, 0/255, 0/255, 1] # unclassified 
    rgba[data==2, :] = [225/255, 175/255, 0/255, 1]
    rgba[data==3, :] = [75/255, 180/255, 210/255, 1]
    rgba[data==4, :] = [75/255, 130/255, 0/255, 1]
    rgba[data==5, :] = [170/255, 50/255, 90/255, 1]
    return rgba

# def correlation_matrix(correlation):
#     plt.matshow(correlation, cmap='cividis') # viridis cividis
#     plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=-55)
#     plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=30)
#     cb = plt.colorbar()
#     cb.ax.tick_params(labelsize=10)
#     plt.title('Correlation Matrix', fontsize=12);
#     plt.show()
#     return None
    
def plotImage(image,labels,cmap,title):
    #-- add legend: https://bitcoden.com/answers/how-to-add-legend-to-imshow-in-matplotlib
    plt.figure()
    plt.imshow(image)
    plt.grid(False)
    plt.title(title)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
    # plt.legend(handles=patches, loc=4)
    plt.legend(handles=patches,loc='lower center', bbox_to_anchor=(0.5, -0.2),
          fancybox=True, shadow=True, ncol=3, prop={'size': 8})
    plt.show()
    return None
    
def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def log_output(in_string):
    f = open(log_file, 'a')
    f.write(in_string)
    f.close()
    return None

# def plotLearningCurve(train_mean, train_std, test_mean, test_std, curve_title):
#     plt.plot(train_size_abs, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
#     plt.fill_between(train_size_abs, train_mean + train_std, train_mean - train_std, alpha=0.3, color='blue')
#     plt.plot(train_size_abs, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
#     plt.fill_between(train_size_abs, test_mean + test_std, test_mean - test_std, alpha=0.3, color='green')
#     plt.title(curve_title)
#     plt.xlabel('Training Data Size')
#     plt.ylabel('Model accuracy (f1 weighted)')
#     plt.grid()
#     plt.legend(loc='lower right')
#     plt.show()

# %% - prediction

for mask, predict_raster in enumerate(test_rasters):
    print(f'Mask number: {mask}')
    #-- Every cell location in a raster has a value assigned to it. 
    #-- When information is unavailable for a cell location, the location will be assigned as NoData. 
    upper_limit = np.finfo(np.float32).max/10
    lower_limit = np.finfo(np.float32).min/10
    
    features_list = []
    pr = rasterio.open(predict_raster)
    out_meta = pr.meta
    pr = pr.read().transpose(1,2,0)
    
    for i, __ in enumerate(feature_list): # need to make into a function
        ft = pr[:,:,i]
        ft[ft == -9999.] = 0.
        ft = np.nan_to_num(ft)
        ft[(ft < lower_limit) | (ft > upper_limit)] = 0.
        # ft = median_filter(ft,size=3)
        # ft[(ft < np.mean(ft) - stdevs * np.std(ft)) | (ft > np.mean(ft) + stdevs * np.std(ft))] = 0. # set anything 3 std from mean to 0
        features_list.append(scaleData(ft))
    
    features_arr = np.array(features_list)
    predictors = np.moveaxis(features_arr,0,-1) # np.ndstack is slow  
    mask = predictors[:,:,0]
    print(f'Read raster to predict: {predict_raster}')
    
    start_time = time.time()
    
    for pkl in test_models: # os.listdir(use_models)
        model = pickle.load(open(pkl, 'rb'))
        print(f'Loaded model: {model}')
        
        rc = np.argwhere(mask>0) # return the rows and columns of array elements that are not zero 
        X_new = predictors[rc[:,0],rc[:,1],:] # return the pixel values of n channels 
        X_new = np.nan_to_num(X_new)
        im_predicted = np.zeros((mask.shape))
        
        print('Predicting...')
        predicted = model.predict(X_new)
        im_predicted[rc[:,0],rc[:,1]] = predicted
        
        # print('Median filter...')
        # im_predicted = median_filter(im_predicted, size=5, mode='reflect')
        
        plot_title = pkl.split('_model')[0]
        
        print('Plotting...')
        plotImage(tf_colorMap(im_predicted),tf_labels,tf_cmap,plot_title)
        
        model = None
        
        print('--Prediction elapsed time: %.3f seconds ---' % (time.time() - start_time))
        
        # if write_prediction:
        #     out_meta.update({"driver": "GTiff",
        #                       "height": predictors.shape[0],
        #                       "width": predictors.shape[1],
        #                       "count": 1})
            
        #     with rasterio.open(prediction_path, "w", **out_meta) as dest:
        #         dest.write(im_predicted, 1)
            
        #     predictors = None
        #     dest = None
                
        #     print(f'Saved prediction raster: {prediction_path}')

    # intersection over union
    if Perform_IOU:
        print(f'\nPerforming intersection over union analysis on {test_masks[mask]}')
        bandmask = rasterio.open(test_masks[mask]).read(1)
        iou_score = jaccard_score(bandmask.ravel(), im_predicted.ravel(), average='macro') # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
        print(f'--IOU: {iou_score:.3f}')
        
        differences = np.where(bandmask < im_predicted, 5, bandmask + im_predicted)
        differences = np.where(bandmask > im_predicted, 3, differences)
        
        false_positives = np.count_nonzero(differences == 5)
        false_negatives = np.count_nonzero(differences == 3)
        print(f'\nNumber of false positives: {false_positives:,}')
        print(f'Percentage of false positives: {(false_positives/differences.size)*100:.2f}')
        print(f'Number of false_negatives: {false_negatives:,}')
        print(f'Percentage of false_negatives: {(false_negatives/differences.size)*100:.2f}\n')
        
        plot_title = 'IOU'
        plotImage(iou_colorMap(differences),iou_labels,iou_cmap,plot_title)
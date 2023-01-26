# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:43:02 2023

@author: sharrm
"""

import datetime
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
from pytictoc import TicToc
from scipy.ndimage import median_filter
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

t = TicToc()
t.tic()


# %% - globals

current_time = datetime.datetime.now()
scale = MinMaxScaler()

#-- When information is unavailable for a cell location, the location will be assigned as NoData. 
upper_limit = np.finfo(np.float32).max/10
lower_limit = np.finfo(np.float32).min/10

feature_list = ["w492nm",               #1
                "w560nm",               #2   
                "w665nm",               #3  
                "w833nm",               #4
                "pSDBg",                #5
                "pSDBg_curvature",      #6
                "pSDBg_roughness",      #7  
                "pSGBg_slope",          #8
                "pSDBg_stdev_slope",    #9
                "pSDBg_tri_Wilson",     #10
                "pSDBr"]                #11

labels = {0: 'U', 1: 'F', 2:'T'}

cmap = {0:[225/255, 245/255, 255/255, 1],
        1:[225/255, 175/255, 0/255, 1],
        2:[75/255, 130/255, 0/255, 1]}


# %% - case

# training ------------
# train = True
# predict = False
# write_model = True
model_dir = r'P:\Thesis\Models'
model_name = model_dir + '\RF_model_11Band_2Masks_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
write_model = False

# prediction ------------
train = False
predict = True

if predict:
    # predict_raster = r"P:\Thesis\Test Data\Portland\_Bands_11Band\_Composite\Portland_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\RockyHarbor\_Bands_11Band\_Composite\RockyHarbor_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\GreatLakes\_Bands_11Band\_Composite\GreatLakes_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\NWHI\_Bands_11Band\_Composite\NWHI_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\FL Keys\_Bands_11Band\_Composite\FL Keys_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\HalfMoon\_Bands_11Band\_Composite\HalfMoon_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\Puerto Real\_Bands_11Band\_Composite\PuertoReal_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\WakeIsland\_Bands_11Band\_Composite\WakeIsland_composite.tif"
    predict_raster = r"P:\Thesis\Test Data\StCroix\_Bands_11Band\_Composite\StCroix_composite.tif"
    # use_model = r"P:\Thesis\Models\RF_model_11Band_Mask_20230123_1422.pkl"
    use_model = r"P:\Thesis\Models\RF_model_11Band_2Masks_20230125_1318.pkl"
    
    write_prediction = True
    # write_prediction = False
    
    if write_prediction:
        prediction_path = os.path.abspath(os.path.join(os.path.dirname(predict_raster), '..', '_Prediction'))
        
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        
        prediction_path = prediction_path + '\prediction_11Band_Mask_' + current_time.strftime('%Y%m%d_%H%M') + '.tif'

# %% - training inputs
flkeys_composite = r"P:\Thesis\Training\FLKeys\_Bands_11Band\_Composite\FLKeys_composite.tif"
keylargo_composite = r"P:\Thesis\Training\KeyLargo\_Train\_Bands_11Band\_Composite\KeyLargo_composite.tif"
flkeys_training_mask = r"P:\Thesis\Samples\Raster\FLKeys_Training.tif"
keylargo_training_mask = r"P:\Thesis\Samples\Raster\KeyLargoExtent_Training.tif"

composite_rasters = [keylargo_composite, flkeys_composite]
training_rasters = [keylargo_training_mask, flkeys_training_mask]


# %% - functions

def colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [225/255, 245/255, 255/255, 1] # unclassified 
    rgba[data==1, :] = [225/255, 175/255, 0/255, 1]
    rgba[data==2, :] = [75/255, 130/255, 0/255, 1]
    return rgba

def correlation_matrix(correlation):
    plt.matshow(correlation, cmap='cividis') # viridis cividis
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=8, rotation=-80)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=8, rotation=30)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=12);
    plt.show()
    
def plotImage(image,labels,cmap):
    #-- add legend: https://bitcoden.com/answers/how-to-add-legend-to-imshow-in-matplotlib
    plt.figure()
    plt.imshow(image)
    plt.grid(False)
    plt.title('Truthiness Prediction')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]
    plt.legend(handles=patches, loc=2)
    plt.show()
    
def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# %% - prepare training data
if train:
    all_predictors = []
    
    for tif in composite_rasters:
        bands = rasterio.open(tif).read().transpose((1,2,0))
        
        predictors_list = []
        
        for i, __ in enumerate(feature_list):
            ft = bands[:,:,i]
            ft[ft == -9999.] = 0.
            ft = np.nan_to_num(ft)
            ft[(ft < lower_limit) | (ft > upper_limit)] = 0.
            ft[(ft < np.mean(ft) - 3 * np.std(ft)) | (ft > np.mean(ft) + 3 * np.std(ft))] = 0. # set anything 3 std from mean to 0
            predictors_list.append(scaleData(ft))
        
        predictors = np.array(predictors_list).transpose((1,2,0))
        bands_t = [predictors[:,:,i].ravel() for i in range(predictors.shape[2])]
        all_predictors.append(np.array(bands_t).transpose())
        # should likely implement this here instead
        # rc = np.argwhere(mask>0) # return the rows and columns of array elements that are not zero 
        # X_new = predictors[rc[:,0],rc[:,1],:] # return the pixel values of n channels 
        # X_new = np.nan_to_num(X_new)
        print(f'Added {tif} to training data set.')
        
        bands = None
    
    all_truthiness = []
    
    for tif in training_rasters:
        bands = rasterio.open(tif).read().transpose((1,2,0))
        tf = bands.ravel()
        all_truthiness.append(tf)
        print(f'Added {tif} to truthiness training data set.')
        
        bands = None
        
    X_train = np.vstack(all_predictors)
    Y_train = np.concatenate(all_truthiness)
    
    # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
    
    print('\nPrepared training data.')


# %% - train/predict

if train:
    # train random forest model
    print('Training model...')
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)     
    model.fit(X_train, Y_train)
    
    if write_model:
        pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
        print(f'Saved random forest model: {model_name}\n')

    feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
    print(f'Feature Importance:\n{feature_importance}')
    
    df = pd.DataFrame(X_train, columns=feature_list)
    correlation = df.corr()
    correlation_matrix(correlation)
elif predict:
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
        ft[(ft < np.mean(ft) - 3 * np.std(ft)) | (ft > np.mean(ft) + 3 * np.std(ft))] = 0. # set anything 3 std from mean to 0
        features_list.append(scaleData(ft))
    
    features_arr = np.array(features_list)
    predictors = np.moveaxis(features_arr,0,-1) # np.ndstack is slow  
    mask = predictors[:,:,0]
    print('Read raster to predict...')
    
    model = pickle.load(open(use_model, 'rb'))
    print(f'Loaded model: {use_model}')
    
    rc = np.argwhere(mask>0) # return the rows and columns of array elements that are not zero 
    X_new = predictors[rc[:,0],rc[:,1],:] # return the pixel values of n channels 
    X_new = np.nan_to_num(X_new)
    im_predicted = np.zeros((mask.shape))
    
    print('Predicting...')
    predicted = model.predict(X_new)
    im_predicted[rc[:,0],rc[:,1]] = predicted
    
    # print('Median filter...')
    # im_predicted = median_filter(im_predicted, size=5, mode='reflect')
    
    print('Plotting...')
    plotImage(colorMap(im_predicted),labels,cmap)  
    
    if write_prediction:
        out_meta.update({"driver": "GTiff",
                          "height": predictors.shape[0],
                          "width": predictors.shape[1],
                          "count": 1})
        
        with rasterio.open(prediction_path, "w", **out_meta) as dest:
            dest.write(im_predicted, 1)
        
        predictors = None
        dest = None
            
        print(f'Saved prediction raster: {prediction_path}')

# # 11:09:47 From  Michael Olsen  to  Everyone:
# # 	https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm
# # 11:10:45 From  Michael Olsen  to  Everyone:
# # 	https://www.esri.com/training/catalog/5eb18cf2a7a78b65b7e26134/deep-learning-using-arcgis/

t.toc()
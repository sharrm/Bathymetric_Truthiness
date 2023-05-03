# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:45:26 2023

@author: sharrm
"""

import datetime
import geopandas as gpd
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from pytictoc import TicToc 
import rasterio
# from sklearn import metrics 
from scipy.ndimage import median_filter
# from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
# from tensorflow import keras # in the future will need to figure out hdf5 
# https://stackoverflow.com/questions/45411700/warning-hdf5-library-version-mismatched-error-python-pandas-windows

tt = TicToc()
# %% - globals

current_time = datetime.datetime.now()
scale = MinMaxScaler()

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
write_model = False

model_dir = r'P:\Thesis\Models' + '\\'
model_name = model_dir + 'RF_model_woutRGBNIRr_nooutlier_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'

# prediction ------------
train = False
predict = True

if predict:
    # predict_raster = r"P:\Thesis\Test Data\Portland\_Bands_11Band\_Composite\Portland_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\RockyHarbor\_Bands_11Band\_Composite\RockyHarbor_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\GreatLakes\_Bands_11Band\_Composite\GreatLakes_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\NWHI\_Bands_11Band\_Composite\NWHI_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\FL Keys\_Bands_11Band\_Composite\FL Keys_composite.tif"
    predict_raster = r"P:\Thesis\Test Data\HalfMoon\_Bands_11Band\_Composite\HalfMoon_composite.tif"
    # predict_raster = r
    use_model = r"P:\Thesis\Models\RF_model_11Band_Mask_20230123_1422.pkl"
    
    # write_prediction = True
    write_prediction = False
    
    if write_prediction:
        prediction_path = os.path.abspath(os.path.join(os.path.dirname(predict_raster), '..', '_Prediction'))
        
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        
        prediction_path = prediction_path + '\prediction_11Band_Mask_' + current_time.strftime('%Y%m%d_%H%M') + '.tif'
        

# %% - training files

# shapefiles
key_largo100_training = r"P:\Thesis\Training\KeyLargo\KeyLargo100.shp"
key_largo150_training = r"P:\Thesis\Training\KeyLargo\KeyLargo150.shp"
puerto_real_training = r"P:\Thesis\Training\PuertoReal\_Test\Forest2.shp"
portland_training = r"P:/Thesis/Samples/Portland300.shp"

shapefiles = [key_largo100_training, key_largo150_training] # list of shapefiles

# raster
# key_largo_raster = r"P:\Thesis\Training\KeyLargo\_Train\_Bands\_Composite\key_largo_composite.tif"
# puerto_real_raster = r"P:\Thesis\Training\PuertoReal\_Train\_Bands\_Composite\puerto_real_composite.tif"
# portland_raster = r"P:\Thesis\Training\Portland\_Bands\_Composite\portland_composite.tif"
# key_largo_raster = r"P:\Thesis\Training\KeyLargo\_Train\_Bands_11Band\_Composite\KeyLargo_composite.tif"
# puerto_real_raster = r"P:\Thesis\Training\PuertoReal\_Train\_Bands_11Band\_Composite\PuertoReal_composite.tif"
# # portland_raster = r"P:\Thesis\Training\Portland\_Bands_11Band\_Composite\Portland_composite.tif"
# key_largo_raster = r"P:\Thesis\Training\KeyLargo\_Train\_Bands_woutRGBNIR\_Composite\KeyLargo_composite.tif"
# puerto_real_raster = r"P:\Thesis\Training\PuertoReal\_Train\_Bands_woutRGBNIR\_Composite\PuertoReal_composite.tif"
key_largo_raster = r'P:/Thesis/Training/KeyLargo/_Train/_Bands_woutRGBNIRr/_Composite/_Train_composite.tif'

rasters = [key_largo_raster]


# %% - functions

def colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [225/255, 245/255, 255/255, 1] # unclassified 
    rgba[data==1, :] = [225/255, 175/255, 0/255, 1]
    rgba[data==2, :] = [75/255, 130/255, 0/255, 1]
    return rgba

def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def check_bounds(src, raster, shapefiles):
    # check if shapefiles point locations are inside the bounds of the raster
    shps = []
    
    for s in shapefiles:
        shp = gpd.read_file(s)
        pts = shp[['Easting_m', 'Northing_m']].to_numpy() # read .shp
        bounds = src.bounds # get raster bounds
        
        # check bounds
        eastings_within = np.logical_and(pts[:,0] > bounds[0], pts[:,0] < bounds[2])
        northings_within = np.logical_and(pts[:,1] > bounds[1], pts[:,1] < bounds[3])
        
        if np.all([eastings_within, northings_within]):
            print(f'Shapefile "{os.path.basename(s)}" within bounds of {os.path.basename(raster)}\n')
            shps.append(shp)
        
    return True, shps

# Sample the raster at every point location and store values in DataFrame; return training data
def get_raster_values(src, shp):
    # Read points from shapefile
    
    X_train = []
    Y_train = []
    
    for i in shp:
        pts = i.loc[:, ('Easting_m', 'Northing_m', 'Truthiness')]
        pts.index = range(len(pts))
        coords = [(x,y) for x, y in zip(pts.Easting_m, pts.Northing_m)]
        
        # https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations
        pts['sample'] = [x for x in src.sample(coords)]
        sample_values = pd.DataFrame(pts['sample'].to_list(), columns=feature_list).to_numpy()
        # sample_values[(sample_values < np.mean(sample_values) - 3 * np.std(sample_values)) | 
        #             (sample_values > np.mean(sample_values) + 3 * np.std(sample_values))] = 0
        
        X_train.append(scale.fit_transform(sample_values))

        # X_train.append(scale.fit_transform(pd.DataFrame(pts['sample'].to_list(), columns=feature_list).to_numpy()))
        Y_train.append(pts['Truthiness'].to_numpy())

    X_train = np.vstack(X_train)
    Y_train = np.concatenate(Y_train)
    
    return X_train, Y_train

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


# %% - predictors

if train:
    # loop through each raster, check .shp within the bounds of raster, 
    # and get raster values at each point
    X_train = []
    Y_train = []
    
    for raster in rasters:
        # open raster
        src = rasterio.open(raster)
        match_bounds, shp = check_bounds(src, raster, shapefiles)
           
        if match_bounds:
            predictors, truthiness = get_raster_values(src, shp)
            
            X_train.append(predictors)
            Y_train.append(truthiness)
            
        src, shp = None, None
    
    # https://gis.stackexchange.com/questions/408854/extract-multiband-raster-values-to-point-shapefile
    
    X_train = np.vstack(X_train)
    Y_train = np.concatenate(Y_train)
    # Y_train = LabelEncoder().fit_transform(Y_train)


# %% - train / predict

if train:
    # train random forest model
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)  # originally 100 and 42
    model.fit(X_train, Y_train)
    if write_model:
        pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
    
    print(f'Saved random forest model: {model_name}\n')

    feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)

    print(f'Feature Importance:\n{feature_importance}')
    
    df = pd.DataFrame(X_train, columns=feature_list)
    correlation = df.corr()
    correlation_matrix(correlation)

# elif predict:
#     # predict using existing model
#     predictors = rasterio.open(predict_raster)
#     out_meta = predictors.meta
#     predictors = predictors.read().transpose(1,2,0)
#     for i in range(predictors.shape[2]):
#         predictors[predictors[:,:,i] == -9999.] = 0.
#         predictors[predictors[:,:,i] == np.nan] = 0.
        
#     mask = predictors[:,:,0]
    
#     print('Read raster to predict...')
    
#     model = pickle.load(open(use_model, 'rb'))
    
#     print(f'Loaded model: {use_model}')
    
#     rc = np.argwhere(mask>0) # return the rows and columns of array elements that are not zero 
#     X_new = predictors[rc[:,0],rc[:,1],:] # return the pixel values of n channels 
    
#     X_new[(X_new < np.mean(X_new) - 3 * np.std(X_new)) | 
#                 (X_new > np.mean(X_new) + 3 * np.std(X_new))] = 0
#     X_new = scale.fit_transform(X_new)
#     X_new = np.nan_to_num(X_new)
    
#     print('Predicting...')
    
#     im_predicted = np.zeros((mask.shape))
#     predicted = model.predict(X_new)
#     im_predicted[rc[:,0],rc[:,1]] = predicted
#     # prediction = np.where(predicted == 'T', 1, 0)
#     # im_predicted[rc[:,0],rc[:,1]] = prediction + 1
    
#     print('Median filter...')
    
#     im_predicted = median_filter(im_predicted, size=11, mode='reflect')
    
#     print('Plotting...')
    
#     plotImage(colorMap(im_predicted),labels,cmap)  
    
#     if write_prediction:
#         out_meta.update({"driver": "GTiff",
#                           "height": predictors.shape[0],
#                           "width": predictors.shape[1],
#                           "count": 1})
        
#         with rasterio.open(prediction_path, "w", **out_meta) as dest:
#             dest.write(im_predicted, 1)
        
#         predictors = None
#         dest = None
            
#         print(f'Saved prediction raster: {prediction_path}')
    
elif predict:
    #-- Every cell location in a raster has a value assigned to it. 
    #-- When information is unavailable for a cell location, the location will be assigned as NoData. 
    upper_limit = np.finfo(np.float32).max/10
    lower_limit = np.finfo(np.float32).min/10
    
    features_list = []
    pr = rasterio.open(predict_raster)
    out_meta = pr.meta
    pr = pr.read().transpose(1,2,0)
    
    for i, __ in enumerate(feature_list):
        ft = pr[:,:,i]
        ft[ft == -9999.] = 0.
        ft[ft == np.nan] = 0.
        ft[(ft < lower_limit) | (ft > upper_limit)] = 0.
        ft[(ft < np.mean(ft) - 3 * np.std(ft)) | (ft > np.mean(ft) + 3 * np.std(ft))] = 0. # set anything 3 std from mean to 0
        # features_list.append(scale.fit_transform(ft))
        features_list.append(ft)
    
    features_arr = np.array(features_list)
    predictors = np.moveaxis(features_arr,0,-1) # np.ndstack is slow  
    mask = predictors[:,:,0]
    
    print('Read raster to predict...')
    
    model = pickle.load(open(use_model, 'rb'))
    
    print(f'Loaded model: {use_model}')
    
    rc = np.argwhere(mask>0) # return the rows and columns of array elements that are not zero 
    X_new = predictors[rc[:,0],rc[:,1],:] # return the pixel values of n channels 
    X_new = np.nan_to_num(X_new)
    print('Predicting...')
    im_predicted = np.zeros((mask.shape))
    predicted = model.predict(X_new)
    im_predicted[rc[:,0],rc[:,1]] = predicted
    # prediction = np.where(predicted == 'T', 1, 0)
    # im_predicted[rc[:,0],rc[:,1]] = prediction + 1
    
    # print('Median filter...')
    
    # im_predicted = median_filter(im_predicted, size=11, mode='reflect')
    
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

tt.toc()

# print("--- Total elapsed time: %.3f seconds ---" % (time.time() - start_time))
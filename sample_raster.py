# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:45:26 2023

@author: sharrm
"""

#https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations

import datetime
import geopandas as gpd
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
# from sklearn import metrics 
# from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
# from tensorflow import keras # in the future will need to figure out hdf5 
# https://stackoverflow.com/questions/45411700/warning-hdf5-library-version-mismatched-error-python-pandas-windows

# %% - case

# train = True
# predict = False

train = False
predict = True

if predict:
    predict_raster = r"P:\Thesis\Test Data\OldOrchard\_Bands\_Composite\old_orchard_composite.tif"
    use_model = r"P:\Thesis\Models\RF_model_20230117_1603.pkl"
    
    write_prediction = False
    
    if write_prediction:
        prediction_path = os.path.abspath(os.path.join(os.path.dirname(predict_raster), '..', '_Prediction'))
        
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        
        prediction_path = prediction_path + '\prediction.tif'
        

# %% - training files

# shapefiles
key_largo_training = r"P:\Thesis\Training\KeyLargo\KeyLargo100.shp"
puerto_real_training = r"P:\Thesis\Training\PuertoReal\_Test\Forest2.shp"

shapefiles = [key_largo_training, puerto_real_training] # list of shapefiles

# raster
key_largo_raster = r"P:\Thesis\Test Data\KeyLargo_Composite_9_compressionless.tif"
puerto_real_raster = r"P:\Thesis\Training\PuertoReal\_Composites\PuertoReal_Composite_9p.tif"

rasters = [key_largo_raster, puerto_real_raster]


# %% - model

current_time = datetime.datetime.now()
model_dir = r'P:\Thesis\Models' + '\\'
model_name = model_dir + 'RF_model_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'



# %% - globals

feature_list = ["w492nm",               #1
                "w560nm",               #2   
                "w665nm",               #3  
                "w833nm",               #4
                "pSDBg",                #5
                "pSDBr",                #6
                "pSDBg_slope",          #7  
                "stdev7_slope7",        #8
                "pSDBg_curve"]          #9

labels = {0: 'U', 1: 'F', 2:'T'}

cmap = {0:[225/255, 245/255, 255/255, 1],
        1:[150/255, 240/255, 100/255, 1],
        2:[75/255, 130/255, 0/255, 1]}


# %% - functions

def colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [225/255, 245/255, 255/255, 1] # unclassified 
    rgba[data==1, :] = [150/255, 240/255, 100/255, 1]
    rgba[data==2, :] = [75/255, 130/255, 0/255, 1]
    return rgba

def scaleData(data):
    # print(np.min(data), np.max(data))
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def check_bounds(src, raster, shapefiles):
    # check if shapefiles point locations are inside the bounds of the raster
    for s in shapefiles:
        shp = gpd.read_file(s)
        pts = shp[['Easting_m', 'Northing_m']].to_numpy() # read .shp
        bounds = src.bounds # get raster bounds
        
        # check bounds
        eastings_within = np.logical_and(pts[:,0] > bounds[0], pts[:,0] < bounds[2])
        northings_within = np.logical_and(pts[:,1] > bounds[1], pts[:,1] < bounds[3])
        
        if np.all([eastings_within, northings_within]):
            print(f'Shapefile "{os.path.basename(s)}" within bounds of {os.path.basename(raster)}\n')
            
            return True, shp

# Sample the raster at every point location and store values in DataFrame; return training data
def get_raster_values(src, shp):
    # Read points from shapefile
    pts = shp.loc[:, ('Easting_m', 'Northing_m', 'Truthiness')]
    pts.index = range(len(pts))
    coords = [(x,y) for x, y in zip(pts.Easting_m, pts.Northing_m)]
    pts['sample'] = [x for x in src.sample(coords)]
    
    X_train = pd.DataFrame(pts['sample'].to_list(), columns=feature_list).to_numpy()
    Y_train = pts['Truthiness'].to_numpy()
    
    return X_train, Y_train

def correlation_matrix(correlation):
    plt.matshow(correlation, cmap='cividis') # viridis cividis
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=8, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=8, rotation=30)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
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
    plt.legend(handles=patches, loc=4)
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
    
    X_train[(X_train < np.mean(X_train) - 3 * np.std(X_train)) | 
                (X_train > np.mean(X_train) + 3 * np.std(X_train))] = 0
    X_train = scaleData(X_train)

# %% - train / predict

if train:
    # train random forest model
    model = RandomForestClassifier(n_estimators = 100, random_state = 42) 
    model.fit(X_train, Y_train)
    pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
    
    print(f'Saved random forest model: {model_name}\n')

    feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)

    print(f'Feature Importance:\n{feature_importance}')
    
    df = pd.DataFrame(X_train, columns=feature_list)
    correlation = df.corr()
    correlation_matrix(correlation)

elif predict:
    # predict using existing model
    predictors = rasterio.open(predict_raster)
    out_meta = predictors.meta
    predictors = predictors.read().transpose(1,2,0)
    mask = predictors[:,:,0]
    
    print(f'Read raster to predict: {predictors.shape}, {mask.shape}')
    
    model = pickle.load(open(use_model, 'rb'))
    
    print(f'Loaded model: {use_model}')
    
    rc = np.argwhere(mask>0) # return the rows and columns of array elements that are not zero 
    X_new = predictors[rc[:,0],rc[:,1],:] # return the pixel values of n channels 

    X_new[(X_new < np.mean(X_new) - 3 * np.std(X_new)) | 
                (X_new > np.mean(X_new) + 3 * np.std(X_new))] = 0
    X_new = scaleData(X_new)
    
    X_new = np.nan_to_num(X_new)
    
    im_predicted = np.zeros((mask.shape))
    predicted = model.predict(X_new)
    prediction = np.where(predicted == 'T', 1, 0)
    im_predicted[rc[:,0],rc[:,1]] = prediction + 1
    
    print('Plotting...')
    
    plotImage(colorMap(im_predicted),labels,cmap)  
    
    if write_prediction:
        out_meta.update({"driver": "GTiff",
                          "height": predictors.shape[0],
                          "width": predictors.shape[1],
                          "count": 1})
        
        with rasterio.open(prediction_path, "w", **out_meta) as dest:
            dest.write(im_predicted, 1) # had to specify '1' here for some reason
        
        predictors = None
        dest = None
            
        print(f'Saved prediction raster: {prediction_path}')
    
    
    
    
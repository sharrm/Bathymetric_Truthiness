# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:45:26 2023

@author: sharrm
"""

#https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import rasterio
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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

# %% - functions

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
    
    # image = src.read()
    
    # for f in feature_list:
        # pts[f] = [x[0] for x in src.sample(coords)]
        
    pts['sample'] = [x for x in src.sample(coords)]
        
    # pts_arr = pts.to_numpy()
    # X_train = pts_arr[:,3:]
    # Y_train = pts_arr[:,2]
    
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

# %% - files

# shapefiles
key_largo_training = r"P:\Thesis\Training\KeyLargo\KeyLargo100.shp"
puerto_real_training = r"P:\Thesis\Training\PuertoReal\_Test\Forest2.shp"

shapefiles = [key_largo_training, puerto_real_training] # list of shapefiles

# raster
key_largo_raster = r"P:\Thesis\Test Data\KeyLargo_Composite_9_compressionless.tif"
puerto_real_raster = r"P:\Thesis\Training\PuertoReal\_Composites\PuertoReal_Composite_9p.tif"

rasters = [key_largo_raster, puerto_real_raster]

# %% - predictors

# looping through each raster, read the raster, check .shp within the bounds, 
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

# %% - train / predict

train = True
predict = False

# train = False
# predict = True


if train:
    # train random forest model
    model = RandomForestClassifier(n_estimators = 100, random_state = 42) 
    model.fit(X_train, Y_train)

    feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)

    print(feature_importance)
    
    df = pd.DataFrame(X_train, columns=feature_list)
    correlation = df.corr()
    correlation_matrix(correlation)

if predict:
    # work on predicting something next
    pass
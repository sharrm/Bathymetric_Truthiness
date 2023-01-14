# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:43:02 2023

@author: sharrm
"""

import pandas as pd
import rasterio
import numpy as np
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

feature_list = ["w492nm",               #1
                "w560nm",               #2   
                "w665nm",               #3  
                "w833nm",               #4
                "pSDBg",                #5
                "pSDBr",                #6
                "pSDBg_slope",          #7  
                "stdev7_slope7",        #8
                "pSDBg_curve"]          #9

bands = rasterio.open(r"P:\Thesis\Test Data\KeyLargo_Composite_9_compressionless.tif").read().transpose((1,2,0))
tf = rasterio.open(r"P:\Thesis\Test Data\mbs_truthiness_compressionless.tif").read(1)

bands_t = [bands[:1930,:1888,i].ravel() for i in range(bands.shape[2])]
X_train = np.array(bands_t).transpose()
Y_train = tf.ravel()

# https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python

# train = True
train = False

if train:
    # train random forest model
    model = RandomForestClassifier(n_estimators = 100, random_state = 42) 
    model.fit(X_train, Y_train)

    feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)

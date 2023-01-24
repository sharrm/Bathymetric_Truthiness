# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:43:02 2023

@author: sharrm
"""

import datetime
import numpy as np
import pandas as pd
import pickle
import rasterio
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

write_model = True
current_time = datetime.datetime.now()

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

bands = rasterio.open(r"P:\Thesis\Training\KeyLargo\_Train\_Bands_11Band\_Composite\KeyLargo_composite.tif").read().transpose((1,2,0))
tf = rasterio.open(r"P:\Thesis\Samples\Raster\raster.tif").read(1)
model_dir = r'P:\Thesis\Models'
model_name = model_dir + '\RF_model_11Band_Mask_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'

bands_t = [bands[:,:,i].ravel() for i in range(bands.shape[2])]
X_train = np.array(bands_t).transpose()
X_train[X_train == -9999.] = 0.
Y_train = tf.ravel()

# https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python

train = True
predict = False
# train = False
# predict = True

if train:
    # train random forest model
    print('Training model...')
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)     
    model.fit(X_train, Y_train)
    
    if write_model:
        pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
        print(f'\nWrote RF model: {model_name}')

    feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
    
    print(f'Feature Importance:\n{feature_importance}')

# # 11:09:47 From  Michael Olsen  to  Everyone:
# # 	https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm
# # 11:10:45 From  Michael Olsen  to  Everyone:
# # 	https://www.esri.com/training/catalog/5eb18cf2a7a78b65b7e26134/deep-learning-using-arcgis/

    
    
    
    
    
    
    
    
    
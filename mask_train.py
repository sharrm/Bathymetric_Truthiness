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
# from scipy.ndimage import median_filter
from sklearn import metrics 
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.inspection import DecisionBoundaryDisplay

t = TicToc()
t.tic()

np.random.seed(0)

# %% - globals

current_time = datetime.datetime.now()
log_file = r'C:\_Thesis\_Monday\_Log\log_' + current_time.strftime('%Y%m%d_%H%M') + '.txt'
# scale = MinMaxScaler()

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

# train = True
predict = False
train = False
# predict = True

# flkeys_composite = r"P:\Thesis\Training\FLKeys\_Bands_11Band\_Composite\FLKeys_composite.tif"
# keylargo_composite = r"P:\Thesis\Training\KeyLargo\_Train\_Bands_11Band\_Composite\KeyLargo_composite.tif"
# flkeys_training_mask = r"P:\Thesis\Samples\Raster\FLKeys_Training.tif"
# keylargo_training_mask = r"P:\Thesis\Samples\Raster\KeyLargoExtent_Training.tif"

# composite_rasters = [keylargo_composite, flkeys_composite]
# training_rasters = [keylargo_training_mask, flkeys_training_mask]

composite_rasters = [
                    r"C:\_Thesis\_Monday\X_train\FLKeys_F_Deep_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\FLKeys_F_Turbid_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\FLKeys_T_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\Halfmoon_F_Deep_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\Halfmoon_F_Turbid_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\Halfmoon_T_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\KeyLargo_F_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\KeyLargo_TF_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\NWHI_F_Deep_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\NWHI_T_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\PR_F_Deep_Clean_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\PR_F_Deep_Noise_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\PR_F_Turbid_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\PR_TF_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\StCroix_F_Deep_composite.tif",
                    r"C:\_Thesis\_Monday\X_train\StCroix_T_composite.tif"
                    ]

training_rasters = [
                    r"C:\_Thesis\_Monday\Y_train\FLKeys_F_Deep_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\FLKeys_F_Turbid_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\FLKeys_T_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\Halfmoon_F_Deep_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\Halfmoon_F_Turbid_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\Halfmoon_T_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\KeyLargo_F_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\KeyLargo_TF_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\NWHI_F_Deep_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\NWHI_T_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\PR_F_Deep_Clean_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\PR_F_Deep_Noise_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\PR_F_Turbid_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\PR_TF_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\StCroix_F_Deep_truthiness.tif",
                    r"C:\_Thesis\_Monday\Y_train\StCroix_T_truthiness.tif"
                    ]

# training ------------
prepare = True
# prepare = False
# n_models = True
n_models = False
# RF = True
RF = False
# SVM = True
SVM = False
# write_model = True
write_model = False
model_dir = r'P:\Thesis\Models'
model_name = model_dir + '\SVM_model_11Band_2Masks_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'

# add logging


# %% prediction ------------

if predict:
    # predict_raster = r"P:\Thesis\Test Data\Portland\_Bands_11Band\_Composite\Portland_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\RockyHarbor\_Bands_11Band\_Composite\RockyHarbor_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\GreatLakes\_Bands_11Band\_Composite\GreatLakes_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\NWHI\_Bands_11Band\_Composite\NWHI_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\FL Keys\_Bands_11Band\_Composite\FL Keys_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\HalfMoon\_Bands_11Band\_Composite\HalfMoon_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\Puerto Real\_Bands_11Band\_Composite\PuertoReal_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\WakeIsland\_Bands_11Band\_Composite\WakeIsland_composite.tif"
    # predict_raster = r"P:\Thesis\Test Data\StCroix\_Bands_11Band\_Composite\StCroix_composite.tif"
    predict_raster = r"P:\Thesis\Test Data\GreatLakes\_Bands_11Band\_Composite\GreatLakes_composite.tif"
    use_model = r"P:\Thesis\Models\SVM_model_11Band_2Masks_20230127_1136.pkl"
    # use_model = r"P:\Thesis\Models\RF_model_11Band_2Masks_20230125_1318.pkl"
    
    # write_prediction = True
    write_prediction = False
    
    if write_prediction:
        prediction_path = os.path.abspath(os.path.join(os.path.dirname(predict_raster), '..', '_Prediction'))
        
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        
        prediction_path = prediction_path + '\prediction_11Band_Mask_' + current_time.strftime('%Y%m%d_%H%M') + '.tif'


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
if prepare:
    all_predictors = []
    
    # add check for extents matching
    
    for tif in composite_rasters:
        bands = rasterio.open(tif).read().transpose((1,2,0))
        # print(f'{os.path.basename(tif)} shape: {bands.shape}')
        
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
        print(f'Added {tif} to X_train training data set.')
        
        f = open(log_file, 'a')
        f.write(f'\nAdded {tif} to X_train training data set.')
        f.close()
        
        bands = None
    
    all_truthiness = []
    
    print('\n')
    
    for tif in training_rasters:
        bands = rasterio.open(tif).read().transpose((1,2,0))
        # print(f'{os.path.basename(tif)} shape: {bands.shape}')
        tf = bands.ravel()
        all_truthiness.append(tf)
        print(f'Added {tif} to Y_train truthiness training data set.')
        
        f = open(log_file, 'a')
        f.write(f'\nAdded {tif} to Y_train truthiness training data set.')
        f.close()
        
        bands = None
        
    x_train = np.vstack(all_predictors)
    y_train = np.concatenate(all_truthiness)
    
    # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
    
    print('\nSplitting data into training and testing...')
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    print('\nPrepared training data...')
    
    print(f'\nX_train size: {X_train.size}\nY_train size: {Y_train.size}\nX_test size: {X_test.size}\nY_test size: {Y_test.size}')
    print(f'\nX_train shape: {X_train.shape}\nY_train shape: {Y_train.shape}\nX_test shape: {X_test.shape}\nY_test shape: {Y_test.shape}')
    
    row_check = (X_train.shape[0] + X_test.shape[0]) - (Y_train.shape[0] + Y_test.shape[0])
    
    if row_check != 0:
        print('X and Y training/test row number mismatch. Stopping...')
        
        f = open(log_file, 'a')
        f.write('\n\nX and Y training/test row number mismatch. Stopping...')
        f.close()
    else:
        print(f'\nX_train + X_test (row check): {X_train.shape[0] + X_test.shape[0]}')
        print(f'Y_train + Y_test (row check): {Y_train.shape[0] + Y_test.shape[0]}')
        print('Verified number of rows in training data correct.')
        
        f = open(log_file, 'a')
        f.write(f'\n\nX_train + X_test (row check): {X_train.shape[0] + X_test.shape[0]}')
        f.write(f'\nY_train + Y_test (row check): {Y_train.shape[0] + Y_test.shape[0]}')
        f.write('\nVerified number of rows in training data correct.')
        f.close()
        

# %% - train/predict

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

if train and RF:
    # train random forest model
    print('Training model...')
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
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
elif train and SVM:
    # train SVM model
    print('Splitting data into training and testing...')
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    print('Training model...')
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    model = SVC(gamma='auto')
    model.fit(X_train, Y_train)
    
    # add in accuracy accessment from metrics
    #-- accuracy assessment 
    acc1 = metrics.accuracy_score(Y_test, model.predict(X_test))*100.0
    print ("\n-RF Validation Accuracy= %.3f %%" % acc1) 
    
    #-- some sklearn classifiers provide the class probabilities for each data 
    # probability = model.predict_proba(X_test)
    
    # add yellowbrick ROCAUC
    
    if write_model:
        pickle.dump(model, open(model_name, 'wb')) # save the trained SVM model
        print(f'Saved svm model: {model_name}\n')
elif train and n_models:

    names = [
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "Nearest Neighbors"
    ]

    classifiers = [
        SVC(kernel="linear", C=0.025),
        SVC(gamma='auto', C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(3)
    ]
    
    print(f'\nTraining with the following models:\n{names}')
    
    for name, clf in zip(names, classifiers):
        print(f'\nTraining {name} model...')
      
        clf.fit(X_train, Y_train)
        
        model_name = model_dir + '\\' + name + '_model_11Band_2Masks_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
        
        #-- accuracy assessment 
        acc1 = metrics.accuracy_score(Y_test, model.predict(X_test))*100.0
        print (f'\n--{name} Validation Accuracy= {acc1}') 
        f = open(log_file, 'a')
        f.write(f'\nTraining with the following models:\n{names}')
        f.write(f'\nTraining {name} model...')
        f.write(f'\n--{name} Validation Accuracy= {acc1}')
        f.close()
        
        if write_model:
            pickle.dump(model, open(model_name, 'wb')) # save the trained model
            print(f'--Saved {name} model: {model_name}\n')     
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
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
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from yellowbrick.classifier import ROCAUC
# from sklearn.metrics import PrecisionRecallDisplay

t = TicToc()
t.tic()

np.random.seed(42)

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
                # "pSDBg_curvature",      #6
                "pSDBg_roughness",      #7  
                # "pSGBg_slope",          #8
                "pSDBg_stdev_slope",    #9
                # "pSDBg_tri_Wilson",     #10
                "pSDBr"]                #11

labels = {0: 'No Data', 1: 'False', 2:'True'}

cmap = {0:[225/255, 245/255, 255/255, 1],
        1:[225/255, 175/255, 0/255, 1],
        2:[75/255, 130/255, 0/255, 1]}


# %% - case

# train = True
# predict = False
train = False
predict = True

# flkeys_composite = r"P:\Thesis\Training\FLKeys\_Bands_11Band\_Composite\FLKeys_composite.tif"
# keylargo_composite = r"P:\Thesis\Training\KeyLargo\_Train\_Bands_11Band\_Composite\KeyLargo_composite.tif"
# flkeys_training_mask = r"P:\Thesis\Samples\Raster\FLKeys_Training.tif"
# keylargo_training_mask = r"P:\Thesis\Samples\Raster\KeyLargoExtent_Training.tif"

# 8 band
flkeys_composite = r"P:\Thesis\Training\FLKeys\_8Band\_Composite\FLKeys_Training_composite.tif"
keylargo_composite = r"P:\Thesis\Training\KeyLargo\_8Band\_Composite\KeyLargoExtent_composite.tif"
stcroix_composite = r"P:\Thesis\Training\StCroix\_8Band\_Composite\StCroix_Extents_TF_composite.tif"

flkeys_training_mask = r"P:\Thesis\Samples\Raster\FLKeys_Training.tif"
keylargo_training_mask = r"P:\Thesis\Samples\Raster\KeyLargoExtent_Training.tif"
stcroix_training_mask = r"P:\Thesis\Samples\Raster\StCroix_Extents_TF_Training.tif"

composite_rasters = [keylargo_composite, flkeys_composite, stcroix_composite]
training_rasters = [keylargo_training_mask, flkeys_training_mask, stcroix_training_mask ]


# training ------------
# prepare = True
prepare = False
single = True
# n_models = True
n_models = False
# RF = True
RF = False
# kfold = True
kfold = False
# write_model = True
write_model = False
stdevs = 5
model_dir = r'P:\Thesis\Models'
# model_dir = r"C:\_Thesis\_Monday\Models"
model_name = model_dir + '\OutlierTest_RF_model_8Band_nMasks_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'

# add logging


# %% prediction ------------

if predict:
    predict_raster = r"P:\Thesis\Test Data\HalfMoon\_8Band\_Composite\Halfmoon_Extents_composite.tif"
    # o_model = r"P:\Thesis\Models\RF_model_8Band_nMasks_20230201_1340.pkl"
    o_model = r"P:\Thesis\Models\OutlierTest{stdevs}_RF_model_8Band_nMasks_20230203_1121.pkl"
    use_models = r"P:\Thesis\Models\RandomForest"

    # write_prediction = True
    write_prediction = False
    
    if write_prediction:
        prediction_path = os.path.abspath(os.path.join(os.path.dirname(predict_raster), '..', '_Prediction'))
        
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        
        prediction_path = prediction_path + '\MLP_prediction_8Band_Mask_5stdev_' + current_time.strftime('%Y%m%d_%H%M') + '.tif'


# %% - functions

def colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [225/255, 245/255, 255/255, 1] # unclassified 
    rgba[data==1, :] = [225/255, 175/255, 0/255, 1]
    rgba[data==2, :] = [75/255, 130/255, 0/255, 1]
    return rgba

def correlation_matrix(correlation):
    plt.matshow(correlation, cmap='cividis') # viridis cividis
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=-55)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=10, rotation=30)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=12);
    plt.show()
    return None
    
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
    plt.legend(handles=patches,loc='lower center', bbox_to_anchor=(0.5, -0.25),
          fancybox=True, shadow=True, ncol=3, prop={'size': 7})
    plt.show()
    return None
    
def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def log_output(in_string):
    f = open(log_file, 'a')
    f.write(in_string)
    f.close()
    return None


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
            ft[(ft < np.mean(ft) - stdevs * np.std(ft)) | (ft > np.mean(ft) + stdevs * np.std(ft))] = 0. # set anything 3 std from mean to 0
            predictors_list.append(scaleData(ft))
        
        predictors = np.array(predictors_list).transpose((1,2,0))
        bands_t = [predictors[:,:,i].ravel() for i in range(predictors.shape[2])]
        all_predictors.append(np.array(bands_t).transpose())
        # should likely implement this here instead
        # rc = np.argwhere(mask>0) # return the rows and columns of array elements that are not zero 
        # X_new = predictors[rc[:,0],rc[:,1],:] # return the pixel values of n channels 
        # X_new = np.nan_to_num(X_new)
        print(f'Added {tif} to X_train training data set. Shape: {np.array(bands_t).transpose().shape}')
        
        log_output(f'\nAdded {tif} to X_train training data set.')
        
        bands = None
    
    all_truthiness = []
    
    print('\n')
    
    for tif in training_rasters:
        bands = rasterio.open(tif).read().transpose((1,2,0))
        # print(f'{os.path.basename(tif)} shape: {bands.shape}')
        tf = bands.ravel()
        all_truthiness.append(tf)
        print(f'Added {tif} to Y_train truthiness training data set. Shape: {tf.shape}')
        

        log_output(f'\nAdded {tif} to Y_train truthiness training data set.')

        
        bands = None
        
    x_train = np.vstack(all_predictors)
    y_train = np.concatenate(all_truthiness)
    
    # https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python
    
    print('\nSplitting data into training and testing...')
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    print('\nPrepared training data...')
    
    print(f'\nX_train pixels: {x_train.size}\nY_train pixels: {y_train.size}')
    print(f'\nX_train shape: {X_train.shape}\nY_train shape: {Y_train.shape}\nX_test shape: {X_test.shape}\nY_test shape: {Y_test.shape}')
    
    row_check = (X_train.shape[0] + X_test.shape[0]) - (Y_train.shape[0] + Y_test.shape[0])
    
    if row_check != 0:
        print('X and Y training/test row number mismatch. Stopping...')

        log_output('\n\nX and Y training/test row number mismatch. Stopping...')

    else:
        print(f'\nX_train + X_test (row check): {X_train.shape[0] + X_test.shape[0]}')
        print(f'Y_train + Y_test (row check): {Y_train.shape[0] + Y_test.shape[0]}')
        print('--Verified number of rows in training data correct.')
        
        log_output(f'\n\nX_train + X_test (row check): {X_train.shape[0] + X_test.shape[0]}'
                   f'\nY_train + Y_test (row check): {Y_train.shape[0] + Y_test.shape[0]}'
                   f'\n--Verified number of rows in training data correct.'
                   f'\n')

        

# %% - train/predict

# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

if train and single:
    # train random forest model
    start_time = time.time()
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    model = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs=-1, oob_score=True)
    # model = MLPClassifier(max_iter=1000, random_state=42, activation='relu', solver='adam')
    # model = HistGradientBoostingClassifier(random_state=42)
    # model = AdaBoostClassifier(random_state=42)
    
    if kfold:
        print('\n--Performing k-fold cross validation...')
        log_output('\n--Performing k-fold cross validation...')
        
        scores = cross_val_score(model, x_train, y_train, cv=5)
        print(f'\nk-fold cross validation results:\n{scores}')
        print("\n%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        log_output(f'\nk-fold cross validation results:\n{scores}'
                   f'\n{scores.mean():.2f} accuracy with a standard deviation of {scores.std():.2f}')
            
    print('\nTraining model...')
    log_output('\nTraining model...')
    model.fit(X_train, Y_train)
    print('Trained model...')
    
    #-- accuracy assessment    
    oob_error = 1 - model.oob_score_
    print(f'\nOOB Error: {oob_error}')
    
    print('\nComputing Precision, Recall, F1...')
    classification = classification_report(Y_test, model.predict(X_test), labels=model.classes_)
    print(f'Classification Report:\n{classification}')

    # prec_recall_f1 = precision_recall_fscore_support(Y_test, model.predict(X_test))
    # print(f'--Precision, Recall, F1:\n{prec_recall_f1}')
        
    # print('\nComputing precision...')
    # precision = precision_score(Y_test, model.predict(X_test), average='weighted')
    # print (f'--Precision: {precision:.3f}')
    
    # print('\nComputing recall...')
    # recall = recall_score(Y_test, model.predict(X_test), average='weighted')
    # print (f'--Recall: {recall:.3f}')
    
    # print('\nComputing F1...')
    # f1 = f1_score(Y_test, model.predict(X_test), average='weighted')
    # print (f'--F1: {f1:.3f} ')
    
    print('\nCreating confusion matrix...')
    cm = confusion_matrix(Y_test, model.predict(X_test), labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='cividis')
    plt.title('Confusion Matrix')
    plt.show()
    
    print('\nComputing accuracy...')
    acc1 = accuracy_score(Y_test, model.predict(X_test))*100.0
    print (f'--Validation Accuracy= {acc1:.2f} %') 
    
    print('\nPlotting ROC AUC...')
    roc_auc=ROCAUC(model, classes=np.unique(Y_train))
    roc_auc.fit(X_train, Y_train)
    roc_auc.score(X_test, Y_test)
    roc_auc.show()
    
    if RF:
        feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
        print(f'\nFeature Importance:\n{feature_importance}')
        log_output(f'\n--Random forest Validation Accuracy= {acc1:.2f} %'
                   f'\nFeature Importance:\n{feature_importance}')
        
        df = pd.DataFrame(X_train, columns=feature_list)
        correlation = df.corr()
        correlation_matrix(correlation)
    
    if write_model:
        pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
        print(f'\nSaved model: {model_name}\n')

    print('--Total elapsed time: %.3f seconds ---' % (time.time() - start_time))
elif train and n_models:
    
    names = [
        "Random Forest",
        "MLP",
        "AdaBoost",
        "Naive Bayes",
        "Hist Gradient Boosting",
        # "Gaussian Process",
        "Decision Tree",
        "Nearest Neighbors",
        "Gradient Boosting",
        "RBF SVM"]
        # "Linear SVM"
        # "QDA"]

    classifiers = [
        RandomForestClassifier(n_estimators=100, random_state=42),
        MLPClassifier(max_iter=1000, random_state=42)]
        # AdaBoostClassifier(random_state=42),
        # GaussianNB(),
        # HistGradientBoostingClassifier(random_state=42),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(random_state=42),
        # KNeighborsClassifier(3),
        # GradientBoostingClassifier(n_estimators=100, random_state=42),
        # SVC(gamma='auto', random_state=42)]
        # SVC(kernel="linear", C=0.025, random_state=42)
        # QuadraticDiscriminantAnalysis()]
    
    print(f'\nTraining with the following models:\n{names}')
    start_time = time.time()
    
    for name, clf in zip(names, classifiers):
        m = TicToc()
        m.tic()
        print(f'\nBegan training {name} model...')
        # train model
        try:
            clf.fit(X_train, Y_train)
            print(f'Trained {name} model...')
                    
            #-- accuracy assessment 
            acc1 = accuracy_score(Y_test, clf.predict(X_test))*100.0
            print (f'--{name} Validation Accuracy= {acc1:.2f} %') 
            m.toc()
            
            f = open(log_file, 'a')
            f.write(f'\nTraining with the following models:\n{names}')
            f.write(f'\nTraining {name} model...')
            f.write(f'\n--{name} Validation Accuracy= {acc1}')
            f.write('--Total elapsed time: %.3f seconds ---' % (time.time() - start_time))
            f.close()
            
            if write_model:
                model_name = model_dir + '\\' + name + '_model_11Band_' + current_time.strftime('%Y%m%d_%H%M') + '.pkl'
                
                pickle.dump(clf, open(model_name, 'wb')) # save the trained model
                print(f'--Saved trained {name} model to {model_name}\n')
                f = open(log_file, 'a')
                f.write(f'\nSaved trained {name} model to {model_name}')
        except:
            print(f'****Issue encountered training {name}')
            f = open(log_file, 'a')
            f.write(f'\n****Issue encountered training {name}')
            f.close()

if predict:
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
        ft[(ft < np.mean(ft) - stdevs * np.std(ft)) | (ft > np.mean(ft) + stdevs * np.std(ft))] = 0. # set anything 3 std from mean to 0
        features_list.append(scaleData(ft))
    
    features_arr = np.array(features_list)
    predictors = np.moveaxis(features_arr,0,-1) # np.ndstack is slow  
    mask = predictors[:,:,0]
    print('Read raster to predict...')
    
    start_time = time.time()
    
    if n_models:
        for pkl in os.listdir(use_models):
            if'SVM' not in pkl and 'Nearest' not in pkl:
                model = pickle.load(open(os.path.join(use_models, pkl), 'rb'))
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
                plotImage(colorMap(im_predicted),labels,cmap,plot_title)
                
                model = None
                
                print('--Prediction elapsed time: %.3f seconds ---' % (time.time() - start_time))
                
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
    else:
        model = pickle.load(open(o_model, 'rb'))
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
        
        plot_title = os.path.basename(o_model).split('_model')[0]
        
        print('Plotting...')
        plotImage(colorMap(im_predicted),labels,cmap,plot_title)
        
        model = None
        
        print('--Prediction elapsed time: %.3f seconds ---' % (time.time() - start_time))
        
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
# %%

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.datasets import make_moons, make_circles, make_classification
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.svm import SVC

# composite_rasters = [
#                     r"C:\_Thesis\_Monday\X_train\FLKeys_F_Deep_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\FLKeys_F_Turbid_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\FLKeys_T_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\Halfmoon_F_Deep_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\Halfmoon_F_Turbid_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\Halfmoon_T_composite.tif",
#                     # r"C:\_Thesis\_Monday\X_train\KeyLargo_F_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\KeyLargo_TF_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\NWHI_F_Deep_composite.tif",
#                     # r"C:\_Thesis\_Monday\X_train\NWHI_T_composite.tif",
#                     # r"C:\_Thesis\_Monday\X_train\PR_F_Deep_Clean_composite.tif",
#                     # r"C:\_Thesis\_Monday\X_train\PR_F_Deep_Noise_composite.tif",
#                     # r"C:\_Thesis\_Monday\X_train\PR_F_Turbid_composite.tif",
#                     # r"C:\_Thesis\_Monday\X_train\PR_TF_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\StCroix_F_Deep_composite.tif",
#                     r"C:\_Thesis\_Monday\X_train\StCroix_T_composite.tif"
#                     ]

# training_rasters = [
                    # r"C:\_Thesis\_Monday\Y_train\FLKeys_F_Deep_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\FLKeys_F_Turbid_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\FLKeys_T_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\Halfmoon_F_Deep_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\Halfmoon_F_Turbid_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\Halfmoon_T_truthiness.tif",
                    # # r"C:\_Thesis\_Monday\Y_train\KeyLargo_F_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\KeyLargo_TF_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\NWHI_F_Deep_truthiness.tif",
                    # # r"C:\_Thesis\_Monday\Y_train\NWHI_T_truthiness.tif",
                    # # r"C:\_Thesis\_Monday\Y_train\PR_F_Deep_Clean_truthiness.tif",
                    # # r"C:\_Thesis\_Monday\Y_train\PR_F_Deep_Noise_truthiness.tif",
                    # # r"C:\_Thesis\_Monday\Y_train\PR_F_Turbid_truthiness.tif",
                    # # r"C:\_Thesis\_Monday\Y_train\PR_TF_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\StCroix_F_Deep_truthiness.tif",
                    # r"C:\_Thesis\_Monday\Y_train\StCroix_T_truthiness.tif"
                    # ]
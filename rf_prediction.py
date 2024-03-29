# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:25:57 2023

@author: sharrm

Updated: 20Mar2023
"""

import datetime
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
# from pytictoc import TicToc
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score 
from scipy.ndimage import median_filter #, gaussian_filter
from skimage import morphology
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, jaccard_score
from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sys
import time
# from yellowbrick.classifier import ROCAUC


# %% - globals

rcParams['figure.dpi'] = 600 # matplotlib plot dpi
np.random.seed(42) 
random_state = 42
n_jobs = 5 # number of cores to use in sklearn processes

current_time = datetime.datetime.now() # current time for file naming
start_time = time.time() # start time for process tracking
log_file = r"C:\_Thesis\_Logs\test_" + current_time.strftime('%Y%m%d_%H%M') + '.txt' # log file output name

#-- When information is unavailable for a cell location, the location will be assigned as NoData. 
upper_limit = np.finfo(np.float32).max/10
lower_limit = np.finfo(np.float32).min/10

# labels for output plots
tf_labels = {0: 'No Data', 1: 'False', 2: 'True'}
iou_labels = {0: 'No Data', 2: 'True Negative', 3: 'False Negative', 4: 'True Positive', 5: 'False Positive'}

# colormap for true or false bathymetry plot legends
tf_cmap = {0:[225/255, 245/255, 255/255, 1],
           1:[225/255, 175/255, 0/255, 1],
           2:[75/255, 130/255, 0/255, 1]}

# colormap for iou similarity plot legends
iou_cmap = {0:[0/255, 0/255, 0/255, 1],
            2:[225/255, 175/255, 0/255, 1],
            3:[75/255, 180/255, 210/255, 1],
            4:[75/255, 130/255, 0/255, 1],
            5:[170/255, 50/255, 90/255, 1]}


# %% - functions

# colormap for true or false bathymetry plots
def tf_colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [0/255, 0/255, 0/255, 1] # unclassified 
    rgba[data==1, :] = [225/255, 175/255, 0/255, 1]
    rgba[data==2, :] = [75/255, 130/255, 0/255, 1]
    return rgba

# colormap for iou similarity plots
def iou_colorMap(data):
    rgba = np.zeros((data.shape[0],data.shape[1],4))
    rgba[data==0, :] = [0/255, 0/255, 0/255, 1] # unclassified 
    rgba[data==2, :] = [225/255, 175/255, 0/255, 1]
    rgba[data==3, :] = [75/255, 180/255, 210/255, 1]
    rgba[data==4, :] = [75/255, 130/255, 0/255, 1]
    rgba[data==5, :] = [170/255, 50/255, 90/255, 1]
    return rgba

# plot prediction image
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

# scale data between 0 and 1
def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# write print statements to log file
def log_output(in_string):
    f = open(log_file, 'a')
    f.write(in_string)
    f.close()
    return None

# pair composite image with labels
def pair_composite_with_labels(test_rasters, test_labels):
    prediction_list = []
    for comp in test_rasters:
        for label in test_labels:
            if rasterio.open(comp).bounds == rasterio.open(label).bounds:
                prediction_list.append((comp,label))
            else:
                continue
    return prediction_list

# read in geotiff and return the image array, metadata, and boundary
def read_image(image):
    img = rasterio.open(image)
    metadata = img.meta
    bounds = img.bounds
    
    composite_arr = img.read().transpose((1,2,0))
    img = None
    return composite_arr, metadata, bounds

# shapes the image array for prediction
def shape_feature_array(composite_arr):
    features = []
    
    for i, __ in enumerate(range(0, composite_arr.shape[2], 1)):
         ft = composite_arr[:,:,i]
         ft[ft == -9999.] = 0.
         ft = np.nan_to_num(ft)
         ft[(ft < lower_limit) | (ft > upper_limit)] = 0.
         features.append(scaleData(ft))
         
    features_arr = np.array(features)
    features_arr = np.moveaxis(features_arr,0,-1) # np.ndstack is slow
    mask = features_arr[:,:,0] 
    rc = np.argwhere(mask>0) # return the rows and columns of array elements that are not zero 
    X_new = features_arr[rc[:,0],rc[:,1],:] # return the pixel values of n channels 
    X_new = np.nan_to_num(X_new)
    im_predicted = np.zeros((mask.shape))
    return im_predicted, X_new, rc

# checks metadata between testing data and labels match
def check_metadata(x_bounds, y_bounds, x_metadata, y_metadata):
    if x_bounds != y_bounds:
        print('Mismatch between training data and labels boundaries...')
        return False
        if x_metadata['crs'] != y_metadata['crs'] or x_metadata['transform'] != y_metadata['transform']:
            print('Mismatch between training data and training label metadata...')
            return False
            if x_metadata['nodata'] != y_metadata['nodata']:
                print('Mismatch between training data and training labels no data values...')
                return False
    else:
        return True    

# writes a geotiff
def write_raster(raster, img_meta, output_path):
    img_meta.update({"driver": "GTiff",
                      "height": raster.shape[0],
                      "width": raster.shape[1],
                      "count": 1})
    
    with rasterio.open(output_path, "w", **img_meta) as dest:
        dest.write(raster, 1)
    
    dest = None
    print(f'Saved prediction raster: {output_path}')
    return None


# %% - evaluation

# compute intersection over union results
def compute_iou(test_mask, im_predicted, iou_metrics, plot_iou, plot_title, write_iou, prediction_path, metadata):
    print(f'Performing intersection over union analysis on {test_mask}')
    bandmask = rasterio.open(test_mask).read(1)
    
    differences = np.where(bandmask < im_predicted, 5, bandmask + im_predicted)
    differences = np.where(bandmask > im_predicted, 3, differences)
    
    compare = bandmask.flatten()
    
    if iou_metrics:
        print('--Computing Precision, Recall, F1-Score...')
        # classification = classification_report(bandmask.flatten(), im_predicted.flatten())
        # print(f'--Classification Report:\n{classification}')
        prc = precision_score(bandmask.flatten(), im_predicted.flatten(), average='macro')
        print (f'--Precision: {prc:.3f}') 
        rcll = recall_score(bandmask.flatten(), im_predicted.flatten(), average='macro')
        print (f'--Recall: {rcll:.3f}') 
        f1 = f1_score(bandmask.flatten(), im_predicted.flatten(), average='macro')
        print (f'--F1-Score: {f1:.3f}') 
        # log_output(f',{prc:.3f},{rcll:.3f},{f1:.3f},{iou_score:.3f}')
        iou_score = jaccard_score(bandmask.ravel(), im_predicted.ravel(), average='macro') 
        print(f'--Mean IOU: {iou_score:.3f}') # 'macro'
        # iou_score = jaccard_score(bandmask.ravel(), im_predicted.ravel(), average=None) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
        # print(f'--IOU: {iou_score}')
    
    if plot_iou:
        plot_title = plot_title + '_IOU'
        plotImage(iou_colorMap(differences),iou_labels,iou_cmap,plot_title)
    
    if write_iou:
        difference_path = prediction_path.replace('prediction_', 'prediction_diff_')
        write_raster(differences, metadata, difference_path) # add meta 
    return iou_score

def prediction_probability(prediction_path, model, im_prob_predicted, X_new, rc, u_metadata, write_prediction):
    prediction_prob = model.predict_proba(X_new)
    prediction_prob = np.amax(prediction_prob, axis=1) # exactly what I need; feed into imshow()
    im_prob_predicted[rc[:,0],rc[:,1]] = prediction_prob
    
    plt.imshow(im_prob_predicted, cmap='cividis')
    plt.colorbar()
    plt.show()
    
    if write_prediction:
        prediction_path = prediction_path.replace('prediction_', 'probability_')
        write_raster(im_prob_predicted, u_metadata, prediction_path)
        print(prediction_path)

    return None

# use model for prediction
def predict_img(unseen_img, im_predicted, X_new, rc, u_metadata, pkl, model, 
                write_prediction, iou_metrics, plot_iou, perform_iou, write_iou, prob_plot, test_mask,
                morph):
    print('--Predicting...')
    prediction_time = time.time()
    prediction = model.predict(X_new)
    performance = round(prediction.size/(time.time() - prediction_time), -3)
    print(f'****Prediction time: {(time.time() - prediction_time):.3f} {performance:,.0f}')

    im_predicted[rc[:,0],rc[:,1]] = prediction
    
    # insert morphology
    if morph:
        pred_erode = morphology.erosion(im_predicted)
        pred_dilate = morphology.dilation(pred_erode)
        # pred_erode = morphology.erosion(pred_dilate)
        # pred_dilate = morphology.dilation(pred_erode)
        im_predicted = pred_dilate
        
        # med = median_filter(im_predicted, size=3)
        # im_predicted = med
        
    
    print('--Plotting...')
    plot_title = os.path.basename(pkl).split('_202')[0]
    plotImage(tf_colorMap(im_predicted),tf_labels,tf_cmap,plot_title)

    # prediction output
    prediction_path = os.path.abspath(os.path.join(os.path.dirname(unseen_img), '..', '_Prediction'))
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)
    print(unseen_img)
    prediction_path = prediction_path + '\\' + os.path.basename(unseen_img).replace('composite_', 'prediction_')
        
    if write_prediction:             
        write_raster(im_predicted, u_metadata, prediction_path)
    
    if prob_plot:
        prediction_probability(prediction_path, model, im_predicted, X_new, rc, u_metadata, write_prediction)
        
    # intersection over union
    if perform_iou:
        mean_iou = compute_iou(test_mask, im_predicted, iou_metrics, plot_iou, plot_title, write_iou, prediction_path, u_metadata)
    else:
        mean_iou = 0
    return im_predicted.size, mean_iou


# %% - sensitivity

# shapes multi-band image array and returns training data ready for sklearn
def shape_test_array(composite_arr):
    features = []
    
    # for i, __ in enumerate(feature_list):
    for i, __ in enumerate(range(0, composite_arr.shape[2], 1)):
          ft = composite_arr[:,:,i] # take ith band
          ft[ft == -9999.] = 0. # set '-9999.' values to 0.
          ft = np.nan_to_num(ft) # set nan values to 0.
          ft[(ft < lower_limit) | (ft > upper_limit)] = 0. # value precision
          features.append(scaleData(ft)) # scale data and append to features list
         
    x_train = np.array(features).transpose((1,2,0)) # shape features list into array
    x_train = [x_train[:,:,i].ravel() for i in range(x_train.shape[2])] # stack features into columns
    return x_train    

# shapes multiple composites for sklearn training input
def shape_multiple_composites(composite_list):
    all_features = []
    x_metadata = []
    x_bounds = []
    
    for comp in composite_list:
        features, metadata, bounds = read_image(comp[0])
        feature_arr = shape_test_array(features)
    
        all_features.append(np.array(feature_arr).transpose())
        x_metadata.append(metadata)
        x_bounds.append(bounds)
        
        print(f'Added {comp[0]} to X_train truthiness training data set. Shape: {features.shape}')
        log_output(f'\nAdded {comp[0]} to X_train truthiness training data set.')

        features = None
        
    x_train = np.vstack(all_features)
    return x_train, x_metadata, x_bounds

# shapes multiple labels for sklearn
def shape_multiple_labels(label_list):
    all_labels = []
    y_metadata = []
    y_bounds = []
    
    for label in label_list:
        labels, metadata, bounds = read_image(label[1])
        label_arr = labels.ravel()
    
        all_labels.append(np.array(label_arr).transpose())
        y_metadata.append(metadata)
        y_bounds.append(bounds)
        
        print(f'Added {label[1]} to Y_train truthiness training data set. Shape: {labels.shape}')
        log_output(f'\nAdded {label[1]} to Y_train truthiness training data set.')

        labels = None
        
    y_train = np.concatenate(all_labels)
    return y_train, y_metadata, y_bounds

# training accuracy
def assess_accuracy(model, X_test, Y_test):   

    X_test = np.delete(X_test, np.where(Y_test == 0), axis = 0)
    Y_test = np.delete(Y_test, np.where(Y_test == 0), axis=0)
    
    print('\nComputing accuracy...')
    acc1 = accuracy_score(Y_test, model.predict(X_test)) * 100.0
    print (f'--Validation Accuracy: {acc1:.2f} %') 
    prc = precision_score(Y_test, model.predict(X_test), average='macro')
    print (f'--Precision: {prc:.3f}') 
    rcll = recall_score(Y_test, model.predict(X_test), average='macro')
    print (f'--Recall: {rcll:.3f}') 
    f1 = f1_score(Y_test, model.predict(X_test), average='macro')
    print (f'--F1-Score: {f1:.3f}') 
    iou_score = jaccard_score(Y_test, model.predict(X_test), average='macro') # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
    print(f'--IOU: {iou_score:.3f}')
    log_output(f'\n{model.n_estimators},{model.min_samples_leaf},{model.min_samples_split},{model.max_depth}')
    log_output(f',{prc:.3f},{rcll:.3f},{f1:.3f},{iou_score:.3f}')
    
    return iou_score


# %% -- main

def main():
    # inputs
    test_models = [
                    # 492, 560, 665, 833, psdbg, psdbr, osi, roughness x3
                    # r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231007_1020.pkl' # works pretty well *keep

# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231009_1339.pkl' # seeming better for KW, vessels, dark area, but could be better
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231009_1344.pkl' # pretty good
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231009_1433.pkl'
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231009_1456.pkl' # great on everything but KW
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231010_0900.pkl' # looking great, except for clouds at KW and Great Lakes
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231010_1050.pkl'

# closer to final
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231010_1323.pkl'
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231010_1442.pkl'
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231011_1045.pkl', # me, stcroix, fl x2, ponce, lookout, peake
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231011_1115.pkl', # me, stcroix, fl, ponce, lookout # great lakes worse
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231011_1150.pkl', # me, stcroix, fl, ponce, lookout
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231011_1207.pkl', # me, stcroix, flx2, ponce, lookout # not good
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231011_1229.pkl', # me, stcroix, fl, ponce, lookout, peake # probably add fl 2 # final final
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231011_1251.pkl' # me, stcroix, fl, ponce, lookout, peake, barn # worse

# 1514
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231012_1514.pkl'
# test again
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231012_1543.pkl'

# rgb nir osi psdbg psdbgR psdbgS
# r'P:\Thesis\Models\RF_8B_100trees_10leaf_2split_Nonedepth_20231015_1438.pkl'
# rgb nir osi psdbg psdbr psdbgR psdbgS
# r'P:\Thesis\Models\RF_9B_100trees_10leaf_2split_Nonedepth_20231015_1501.pkl'
# rgb nir osi psdbg psdbgR psdbgS psdbr psdbrR
# r'P:\Thesis\Models\RF_10B_100trees_10leaf_2split_Nonedepth_20231015_1516.pkl'
# rgb nir osi psdbg psdbgR psdbgS psdbrR
# r'P:\Thesis\Models\RF_9B_100trees_10leaf_2split_Nonedepth_20231015_1534.pkl'
# rgb nir osi psdbg psdbgR psdbgS psdbrS
# r'P:\Thesis\Models\RF_9B_100trees_10leaf_2split_Nonedepth_20231015_1559.pkl'
# rgb nir osi psdbg
# r'P:\Thesis\Training\_Manuscript_Train\Models\RF_6B_100trees_10leaf_2split_Nonedepth_20231015_1619.pkl'

# rgb nir osi psdbg psdbgR psdbgS psdbrR
# r'P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_200trees_10leaf_2split_Nonedepth_20231015_1703.pkl',
# r'P:\Thesis\Training\_Manuscript_Train\Models\Hist_9B_6In_LR0.2_L20.2_20231015_1703.pkl'

# rgb nir osi psdbg psdbgR psdbgS psdbrR # sensitivity 
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_1leaf_2split_5depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_1leaf_2split_10depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_1leaf_2split_20depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_1leaf_2split_Nonedepth_20231015_1729.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_1leaf_20split_Nonedepth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_1leaf_50split_Nonedepth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_10leaf_2split_Nonedepth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_10leaf_20split_20depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_100leaf_2split_10depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_100leaf_2split_20depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_100leaf_2split_Nonedepth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_50trees_100leaf_20split_20depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_1leaf_2split_5depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_1leaf_2split_10depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_1leaf_2split_20depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_1leaf_2split_Nonedepth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_1leaf_20split_Nonedepth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_1leaf_50split_Nonedepth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_10leaf_20split_20depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_100leaf_2split_10depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_100leaf_2split_20depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_100leaf_2split_Nonedepth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_100trees_100leaf_20split_20depth_20231015_1737.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_9B_1000trees_10leaf_2split_Nonedepth_20231015_1737.pkl",

# rgb nir osi psdbg psdbgR psdbgS # sensitivity 
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_50trees_1leaf_2split_20depth_20231018_0939.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_50trees_1leaf_2split_Nonedepth_20231018_0939.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_50trees_1leaf_20split_Nonedepth_20231018_0939.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_50trees_1leaf_50split_Nonedepth_20231018_0939.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_50trees_5leaf_2split_Nonedepth_20231018_0939.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_100trees_1leaf_2split_20depth_20231018_0939.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_100trees_1leaf_20split_Nonedepth_20231018_0939.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_100trees_1leaf_50split_Nonedepth_20231018_0939.pkl",
r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_100trees_10leaf_2split_Nonedepth_20231018_0939.pkl",
# r"P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_100trees_100leaf_2split_Nonedepth_20231018_0939.pkl",

# all data 8 bands
# r'P:\Thesis\Training\_Manuscript_Train\Models\RF_8B_100trees_10leaf_2split_Nonedepth_20231018_1812.pkl'

                   ]
    
    test_composites = [

# ten close to final
    # 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_10Bands\\_Composite\\GreatLakes_Mask_NoLand_10Bands_composite_20231010_1332.tif',
    # 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_10Bands\\_Composite\\KeyWest_Mask_10Bands_composite_20231010_1332.tif', 
    # 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_10Bands\\_Composite\\CapeCod_Mask_10Bands_composite_20231010_1332.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_10Bands\\_Composite\\Niihua4_10Bands_composite_20231010_1332.tif', 
    # 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_10Bands\\_Composite\\Saipan_Extents_NoIsland_10Bands_composite_20231010_1332.tif'

# rgb
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_3Bands\\_Composite\\GreatLakes_Mask_NoLand_3Bands_composite_20231011_1558.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_3Bands\\_Composite\\KeyWest_Mask_3Bands_composite_20231011_1558.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_3Bands\\_Composite\\CapeCod_Mask_3Bands_composite_20231011_1558.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_3Bands\\_Composite\\Niihua_Mask_3Bands_composite_20231011_1558.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_3Bands\\_Composite\\Saipan_Extents_NoIsland_3Bands_composite_20231011_1558.tif'
# rgb nir
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_4Bands\\_Composite\\GreatLakes_Mask_NoLand_4Bands_composite_20231011_1709.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_4Bands\\_Composite\\KeyWest_Mask_4Bands_composite_20231011_1709.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_4Bands\\_Composite\\CapeCod_Mask_4Bands_composite_20231011_1709.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_4Bands\\_Composite\\Niihua_Mask_4Bands_composite_20231011_1709.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_4Bands\\_Composite\\Saipan_Extents_NoIsland_4Bands_composite_20231011_1709.tif'
# rgb nir osi 
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_5Bands\\_Composite\\GreatLakes_Mask_NoLand_5Bands_composite_20231011_1722.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_5Bands\\_Composite\\KeyWest_Mask_5Bands_composite_20231011_1722.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_5Bands\\_Composite\\CapeCod_Mask_5Bands_composite_20231011_1722.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_5Bands\\_Composite\\Niihua_Mask_5Bands_composite_20231011_1722.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_5Bands\\_Composite\\Saipan_Extents_NoIsland_5Bands_composite_20231011_1722.tif'
# rgb nir osi psdbg 
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_6Bands\\_Composite\\GreatLakes_Mask_NoLand_6Bands_composite_20231011_1725.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_6Bands\\_Composite\\KeyWest_Mask_6Bands_composite_20231011_1725.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_6Bands\\_Composite\\CapeCod_Mask_6Bands_composite_20231011_1725.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_6Bands\\_Composite\\Niihua_Mask_6Bands_composite_20231011_1725.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_6Bands\\_Composite\\Saipan_Extents_NoIsland_6Bands_composite_20231011_1725.tif'
# rgb nir osi psdbg psdbgR psdbgSt
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_8Bands\\_Composite\\GreatLakes_Mask_NoLand_8Bands_composite_20231011_1752.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_8Bands\\_Composite\\KeyWest_Mask_8Bands_composite_20231011_1752.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_8Bands\\_Composite\\CapeCod_Mask_8Bands_composite_20231011_1752.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_8Bands\\_Composite\\Niihua_Mask_8Bands_composite_20231011_1752.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_8Bands\\_Composite\\Saipan_Extents_NoIsland_8Bands_composite_20231011_1752.tif'
# rgb nir osi psdbg psdbr
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_7Bands\\_Composite\\GreatLakes_Mask_NoLand_7Bands_composite_20231012_1040.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_7Bands\\_Composite\\KeyWest_Mask_7Bands_composite_20231012_1040.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_7Bands\\_Composite\\CapeCod_Mask_7Bands_composite_20231012_1040.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_7Bands\\_Composite\\Niihua_Mask_7Bands_composite_20231012_1040.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_7Bands\\_Composite\\Saipan_Extents_NoIsland_7Bands_composite_20231012_1040.tif'
# rgb nir osi psdbg
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_5Bands\\_Composite\\GreatLakes_Mask_NoLand_5Bands_composite_20231012_1053.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_5Bands\\_Composite\\KeyWest_Mask_5Bands_composite_20231012_1053.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_5Bands\\_Composite\\CapeCod_Mask_5Bands_composite_20231012_1053.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_5Bands\\_Composite\\Niihua_Mask_5Bands_composite_20231012_1053.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_5Bands\\_Composite\\Saipan_Extents_NoIsland_5Bands_composite_20231012_1053.tif'
# rgb nir osi psdbg psdbgR
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_7Bands\\_Composite\\GreatLakes_Mask_NoLand_6Bands_composite_20231012_1213.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_7Bands\\_Composite\\KeyWest_Mask_6Bands_composite_20231012_1213.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_7Bands\\_Composite\\CapeCod_Mask_6Bands_composite_20231012_1213.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_7Bands\\_Composite\\Niihua_Mask_6Bands_composite_20231012_1213.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_7Bands\\_Composite\\Saipan_Extents_NoIsland_6Bands_composite_20231012_1213.tif'
# rgb nir osi psdbg psdbgSt
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_7Bands\\_Composite\\GreatLakes_Mask_NoLand_6Bands_composite_20231012_1302.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_7Bands\\_Composite\\KeyWest_Mask_6Bands_composite_20231012_1302.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_7Bands\\_Composite\\CapeCod_Mask_6Bands_composite_20231012_1302.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_7Bands\\_Composite\\Niihua_Mask_6Bands_composite_20231012_1302.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_7Bands\\_Composite\\Saipan_Extents_NoIsland_6Bands_composite_20231012_1302.tif'
# rgb nir osi psdbg psdbgR psdbgSt
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_7Bands\\_Composite\\GreatLakes_Mask_NoLand_7Bands_composite_20231012_1324.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_7Bands\\_Composite\\KeyWest_Mask_7Bands_composite_20231012_1324.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_7Bands\\_Composite\\CapeCod_Mask_7Bands_composite_20231012_1324.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_7Bands\\_Composite\\Niihua_Mask_7Bands_composite_20231012_1324.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_7Bands\\_Composite\\Saipan_Extents_NoIsland_7Bands_composite_20231012_1324.tif'
# rgb nir osi psdbg psdbgR psdbgSt psdbr
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_8Bands\\_Composite\\GreatLakes_Mask_NoLand_8Bands_composite_20231012_1339.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_8Bands\\_Composite\\KeyWest_Mask_8Bands_composite_20231012_1339.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_8Bands\\_Composite\\CapeCod_Mask_8Bands_composite_20231012_1339.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_8Bands\\_Composite\\Niihua_Mask_8Bands_composite_20231012_1339.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_8Bands\\_Composite\\Saipan_Extents_NoIsland_8Bands_composite_20231012_1339.tif'
# rgb nir osi psdbg psdbgR psdbgSt psdbr psdbrR
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_9Bands\\_Composite\\GreatLakes_Mask_NoLand_9Bands_composite_20231012_1358.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_9Bands\\_Composite\\KeyWest_Mask_9Bands_composite_20231012_1358.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_9Bands\\_Composite\\CapeCod_Mask_9Bands_composite_20231012_1358.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_9Bands\\_Composite\\Niihua_Mask_9Bands_composite_20231012_1358.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_9Bands\\_Composite\\Saipan_Extents_NoIsland_9Bands_composite_20231012_1358.tif'
# nir osi psdbg psdbgR psdbgSt psdbr psdbrR
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_7Bands\\_Composite\\GreatLakes_Mask_NoLand_7Bands_composite_20231012_1418.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_7Bands\\_Composite\\KeyWest_Mask_7Bands_composite_20231012_1418.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_7Bands\\_Composite\\CapeCod_Mask_7Bands_composite_20231012_1418.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_7Bands\\_Composite\\Niihua_Mask_7Bands_composite_20231012_1418.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_7Bands\\_Composite\\Saipan_Extents_NoIsland_7Bands_composite_20231012_1418.tif'
# rgb nir osi psdbg psdbgR psdbgSt psdbr
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_9Bands\\_Composite\\GreatLakes_Mask_NoLand_9Bands_composite_20231012_1437.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_9Bands\\_Composite\\KeyWest_Mask_9Bands_composite_20231012_1437.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_9Bands\\_Composite\\CapeCod_Mask_9Bands_composite_20231012_1437.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_9Bands\\_Composite\\Niihua_Mask_9Bands_composite_20231012_1437.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_9Bands\\_Composite\\Saipan_Extents_NoIsland_9Bands_composite_20231012_1437.tif'
# rgb nir osi psdbg psdbgR psdbgSt psdbr psdbrR
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_10Bands\\_Composite\\GreatLakes_Mask_NoLand_10Bands_composite_20231012_1505.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_10Bands\\_Composite\\KeyWest_Mask_10Bands_composite_20231012_1505.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_10Bands\\_Composite\\CapeCod_Mask_10Bands_composite_20231012_1505.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_10Bands\\_Composite\\Niihua_Mask_10Bands_composite_20231012_1505.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_10Bands\\_Composite\\Saipan_Extents_NoIsland_10Bands_composite_20231012_1505.tif'

# rgb nir osi psdbg psdbgR psdbgS *
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_8Bands\\_Composite\\GreatLakes_Mask_NoLand_8Bands_composite_20231015_1445.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_8Bands\\_Composite\\KeyWest_Mask_8Bands_composite_20231015_1445.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_8Bands\\_Composite\\CapeCod_Mask_8Bands_composite_20231015_1445.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_8Bands\\_Composite\\Niihua_Mask_8Bands_composite_20231015_1445.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_8Bands\\_Composite\\Saipan_Extents_NoIsland_8Bands_composite_20231015_1445.tif'
# rgb nir osi psdbg psdbr psdbgR psdbgS
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_9Bands\\_Composite\\GreatLakes_Mask_NoLand_9Bands_composite_20231015_1509.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_9Bands\\_Composite\\KeyWest_Mask_9Bands_composite_20231015_1509.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_9Bands\\_Composite\\CapeCod_Mask_9Bands_composite_20231015_1509.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_9Bands\\_Composite\\Niihua_Mask_9Bands_composite_20231015_1509.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_9Bands\\_Composite\\Saipan_Extents_NoIsland_9Bands_composite_20231015_1509.tif'
# rgb nir osi psdbg psdbgR psdbgS psdbr psdbrR
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_10Bands\\_Composite\\GreatLakes_Mask_NoLand_10Bands_composite_20231015_1526.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_10Bands\\_Composite\\KeyWest_Mask_10Bands_composite_20231015_1526.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_10Bands\\_Composite\\CapeCod_Mask_10Bands_composite_20231015_1526.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_10Bands\\_Composite\\Niihua_Mask_10Bands_composite_20231015_1526.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_10Bands\\_Composite\\Saipan_Extents_NoIsland_10Bands_composite_20231015_1526.tif'
# rgb nir osi psdbg psdbgR psdbgS psdbrR *
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_9Bands\\_Composite\\GreatLakes_Mask_NoLand_9Bands_composite_20231015_1530.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_9Bands\\_Composite\\KeyWest_Mask_9Bands_composite_20231015_1530.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_9Bands\\_Composite\\CapeCod_Mask_9Bands_composite_20231015_1530.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_9Bands\\_Composite\\Niihua_Mask_9Bands_composite_20231015_1530.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_9Bands\\_Composite\\Saipan_Extents_NoIsland_9Bands_composite_20231015_1530.tif'
# rgb nir osi psdbg psdbgR psdbgS psdbrS
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_9Bands\\_Composite\\GreatLakes_Mask_NoLand_9Bands_composite_20231015_1609.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_9Bands\\_Composite\\KeyWest_Mask_9Bands_composite_20231015_1609.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_9Bands\\_Composite\\CapeCod_Mask_9Bands_composite_20231015_1609.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_9Bands\\_Composite\\Niihua_Mask_9Bands_composite_20231015_1609.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_9Bands\\_Composite\\Saipan_Extents_NoIsland_9Bands_composite_20231015_1609.tif'
# rgb nir osi psdbg
# 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\GreatLakes\\_Features_6Bands\\_Composite\\GreatLakes_Mask_NoLand_6Bands_composite_20231015_1625.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\KeyWest\\_Features_6Bands\\_Composite\\KeyWest_Mask_6Bands_composite_20231015_1625.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Mass\\_Features_6Bands\\_Composite\\CapeCod_Mask_6Bands_composite_20231015_1625.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Niihau\\_Features_6Bands\\_Composite\\Niihua_Mask_6Bands_composite_20231015_1625.tif', 'P:\\Thesis\\Test Data\\_Manuscript_Test\\Imagery\\Saipan\\_Features_6Bands\\_Composite\\Saipan_Extents_NoIsland_6Bands_composite_20231015_1625.tif'

r"C:\_Thesis\Data\Testing\CapeCod_Mask_8Bands_composite_20231015_1445.tif",
r"C:\_Thesis\Data\Testing\GreatLakes_Mask_NoLand_8Bands_composite_20231015_1445.tif",
r"C:\_Thesis\Data\Testing\KeyWest_Mask_8Bands_composite_20231015_1445.tif",
r"C:\_Thesis\Data\Testing\Niihua_Mask_8Bands_composite_20231015_1445.tif",
"C:\_Thesis\Data\Testing\Saipan_Extents_NoIsland_8Bands_composite_20231015_1445.tif",

                       ]
    
    test_labels = [ 
                        r'P:\Thesis\Masks\KeyWest_Mask_TF.tif',
                        r'P:\Thesis\Masks\CapeCod_Mask_TF.tif',
                        r"C:\_Thesis\Masks\Test\GreatLakes_Mask_NoLand_TF.tif",
                        r"P:\Thesis\Test Data\_Manuscript_Test\Masks\Niihua_Mask_TF.tif",
                        r"C:\_Thesis\Masks\Test\Saipan_Mask_NoIsland_TF.tif",
                        
                        # r"P:\Thesis\Masks\_Archive\KeyWest_Mask_Deep_TF.tif",
                    ]
    
    prediction_options = {'write_prediction': True,
                          'perform_iou': False,
                          'iou_metrics': False,
                          'plot_iou': False,
                          'write_iou': False,
                          'prob_plot': False,
                          'morph': True
                          }
    
    total_accuracy = True
    # with_mask = False
    with_mask = True
    
    results = []
    
    if with_mask:
        prediction_list = pair_composite_with_labels(test_composites, test_labels)
        
        for unseen_img in prediction_list:
            u_img, u_metadata, u_bounds = read_image(unseen_img[0])
            print(f'Read raster to predict: {unseen_img[0]}')
            log_output(f'Read raster to predict: {os.path.basename(unseen_img[0])}')
            log_output('\nN Trees, Leaf, Split, Depth, Precision, Recall, F1-Score, IOU')
            
            im_predicted, X_new, rc = shape_feature_array(u_img)
            start_iou = 0
            
            for pkl in test_models:
                model = pickle.load(open(pkl, 'rb'))
                print(f'\nLoaded model: {model}')
                # log_output(f'\n{model.n_estimators},{model.min_samples_leaf},{model.min_samples_split},{model.max_depth}')
                prediction_time = time.time()
                img_size, max_iou = predict_img(unseen_img[0], im_predicted, X_new, rc, u_metadata, pkl, model, **prediction_options, test_mask=unseen_img[1])
                model = None
                
                performance = round(img_size/(time.time() - prediction_time), -3)
                
                if max_iou > start_iou:
                    start_iou = max_iou
                    print(f'--**Max IOU: {max_iou:.3f}')
                    top_model = pkl
                else:
                    top_model = 'Not selected'
                
                # log_output(f',{performance:.0f}')
                
                print(f'--Prediction elapsed time: {(time.time() - prediction_time):.1f} ({img_size/(time.time() - prediction_time):,.0f} pixels per second)')
            print(f'**** Max IOU {max_iou:.3f} with {top_model} ')
            results.append((os.path.basename(unseen_img[0]).split('_')[0],os.path.basename(top_model),round(max_iou,4)))
            print('------------------------------------------------------------------------')
            log_output('\n-------------------------------------------------------------------\n')
    elif total_accuracy:
        # total model testing results
        prediction_list = pair_composite_with_labels(test_composites, test_labels)

        x_test, x_metadata, x_bounds = shape_multiple_composites(prediction_list)
        y_test, y_metadata, y_bounds = shape_multiple_labels(prediction_list)
        
        start_iou = 0
        
        for pkl in test_models:
            model = pickle.load(open(pkl, 'rb'))
            print(f'\n--Loaded model: {model}')
            iou_score = assess_accuracy(model, x_test, y_test)
            
            if iou_score > start_iou:
                start_iou = iou_score
                print(f'--**Max IOU: {iou_score:.3f}')
                top_model = pkl
            else:
                top_model = 'Not selected'
                
        print(f'**** Max IOU {iou_score:.3f} with {top_model} ')
    else:
        for unseen_img in test_composites:
            u_img, u_metadata, u_bounds = read_image(unseen_img)
            print(f'Read raster to predict: {unseen_img}')
            
            im_predicted, X_new, rc = shape_feature_array(u_img)
        
            for pkl in test_models:
                model = pickle.load(open(pkl, 'rb'))
                print(f'\n--Loaded model: {model}')
                prediction_time = time.time()
                img_size = predict_img(unseen_img, im_predicted, X_new, rc, u_metadata, pkl, model, **prediction_options, test_mask=None)
                model = None
        
                print(f'--Prediction elapsed time: {(time.time() - prediction_time):.1f} ({img_size/(time.time() - prediction_time):,.0f} pixels per second)')
            print('------------------------------------------------------------------------\n')
    print(f'\nFinal results: {results}')
    return None

    # predict with all input models and images
    # for pkl in test_models:
    #     print(f'Loading {pkl}')
    #     model = pickle.load(open(pkl, 'rb'))
    #     print(f'--Loaded model: {model}')
        
    #     for unseen_img in prediction_list:
    #         predict_img(pkl, model, unseen_img[0], **prediction_options, test_mask=unseen_img[1])

    #     print('--Prediction elapsed time: %.3f seconds ---' % (time.time() - start_time))
    #     print('------------------------------------------------------------------------')
    # return None

if __name__ == '__main__':
    start = current_time.strftime('%H:%M:%S')
    print(f'Starting at {start}\n')
    main()
    runtime = time.time() - start_time
    print(f'\nTotal elapsed time: {runtime:.1f} seconds / {(runtime/60):.1f} minutes')
   

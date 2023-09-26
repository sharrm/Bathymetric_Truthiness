# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:19:30 2023

@author: sharrm


"""

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
from rasterio.mask import mask
from skimage.morphology import binary_dilation, binary_erosion
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier 
from sklearn.metrics import accuracy_score#, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score 
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# %% - morphology

def near_land(input_blue, input_green, input_red, input_704, input_nir, shapefile, out_dir, write):
    
    # Open the geotiff file
    with rasterio.open(input_green) as green:
        # Read the green band metadata
        out_meta = green.meta
        
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_green, transform = mask(green, gdf.geometry, crop=True)
    
    # Open the geotiff file
    with rasterio.open(input_nir) as nir:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_nir, transform = mask(nir, gdf.geometry, crop=True)
        
    # Open the geotiff file
    with rasterio.open(input_704) as b704:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_704, transform = mask(b704, gdf.geometry, crop=True)        
        
    # Open the geotiff file
    with rasterio.open(input_blue) as blue:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_blue, transform = mask(blue, gdf.geometry, crop=True)
            
    # Open the geotiff file
    with rasterio.open(input_red) as red:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_red, transform = mask(red, gdf.geometry, crop=True)            
    
    # compute ndwi
    ndwi = (cropped_green - cropped_nir) / (cropped_green + cropped_nir)
    cropped = np.moveaxis(ndwi, 0, -1)[:,:,0]
    
    # compute pSDBr
    pSDBr = np.log(cropped_blue * 1000) / np.log(cropped_red * 1000)
    pSDBg = np.log(cropped_blue * 1000) / np.log(cropped_green * 1000)  
    
    # create binary array for land and water pixels
    nan_vals = np.where(np.isnan(cropped))
    cropped_land_water = np.where(cropped < 0.15, 1, 0)
    
    # morphological operation to grow land pixels
    morphed_land = binary_dilation(cropped_land_water) #.astype(cropped_land_water.dtype))
    erode_land = binary_erosion(morphed_land) #.astype(cropped_land_water.dtype))
    
    # pixels adjacent to land
    zero_mask = np.logical_and(morphed_land, ~erode_land)
    land_adjacent_ndwi = np.where(zero_mask, cropped, 0)    
    # land_adjacent_ndwi = np.where(land_adjacent_ndwi < 0.15, 0, land_adjacent_ndwi)
    # land_adjacent_percentile = np.where(np.percentile(land_adjacent_ndwi, 90), land_adjacent_ndwi, 0)
    percentile10 = np.nanpercentile(cropped[zero_mask == 1], 10)
    print(f'Precentile 10: {percentile10}')
    percentile10 = np.where(land_adjacent_ndwi < percentile10, land_adjacent_ndwi, 0)
    
    percentile90 = np.nanpercentile(cropped[zero_mask == 1], 90)
    print(f'Precentile 90: {percentile90}')
    percentile90 = np.where(land_adjacent_ndwi > percentile90, land_adjacent_ndwi, 0)
    
    # ndwi values for pixels adjacent to land for histogram
    ndwi_adjacent = cropped[zero_mask == 1]
    print(f'Average land adjacent NDWI value: {np.nanmean(ndwi_adjacent):.3f} ± {np.nanstd(ndwi_adjacent):.3f}')
    print(f'Median land adjacent NDWI value: {np.nanmedian(ndwi_adjacent):.3f}')
    land_adjacent_ndwi[nan_vals] = np.nan
    percentile10[nan_vals] = np.nan
    percentile90[nan_vals] = np.nan
    
    
    def normalize(band):
        band_min, band_max = (np.nanmin(band), np.nanmax(band))
        return ((band-band_min)/((band_max - band_min)))

    red_n = normalize(cropped_red[0,:,:])
    green_n = normalize(cropped_green[0,:,:])
    blue_n = normalize(cropped_blue[0,:,:])
    
    rgb_composite_n = np.dstack((red_n, green_n, blue_n))
    
    # Stack the bands to create an RGB image
    rgb_image = np.dstack((cropped_red[0,:,:], cropped_green[0,:,:], cropped_blue[0,:,:]))
    brightened_image = np.clip(rgb_composite_n * 3, 0, 255)#.astype(np.uint8)
    brightened_image[nan_vals] = 255
    m = np.ma.masked_where(np.isnan(brightened_image),brightened_image)

    
    # plt.figure(figsize=(10, 10))
    f, ax = plt.subplots(2,2, figsize=(10, 6), dpi=200)
    ax[0,0].imshow(brightened_image)
    ax[0,0].set_title('RGB', fontsize=10)
    ax[0,1].imshow(land_adjacent_ndwi, vmax=0.1, cmap='cividis')
    ax[0,1].set_title('Land Adjacent Pixels', fontsize=10)
    ax[1,0].imshow(percentile10, vmax=0.01, cmap='cividis')
    ax[1,0].set_title('10th Percentile', fontsize=10)
    ax[1,1].imshow(percentile90, vmax=0.05, cmap='cividis')
    ax[1,1].set_title('90th Percentile', fontsize=10)
    # plt.tight_layout()
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
    plt.show()
    
    # ndwi values for pixels adjacent to land for histogram
    ndwi_adjacent = cropped[zero_mask == 1]
    print(f'Average land adjacent NDWI value: {np.nanmean(ndwi_adjacent):.3f} ± {np.nanstd(ndwi_adjacent):.3f}')
    print(f'Median land adjacent NDWI value: {np.nanmedian(ndwi_adjacent):.3f}')
    land_adjacent_ndwi[nan_vals] = np.nan
        
    training_data = np.vstack((cropped_blue.flatten(), 
                               cropped_green.flatten(), 
                               cropped_red.flatten(),
                                cropped_704.flatten(),
                               cropped_nir.flatten(), 
                               ndwi.flatten(), 
                               # pSDBg.flatten(),
                               pSDBr.flatten())).transpose()
    training_data[np.isnan(training_data)] = 2
    
    # Plot the masked image
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(cropped, cmap='gray', vmin=0.2)
    plt.title('Land Adjacent Pixels')
    plt.imshow(zero_mask, cmap='Reds', alpha=0.3, vmax=0.2)
    plt.colorbar()
    
    # Plot histogram of values
    plt.subplot(2, 2, 2)
    plt.hist(ndwi_adjacent, bins=50, edgecolor='k')
    plt.xlabel('NDWI Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of NDWI Values at Land Adjacent Pixels')
    plt.tight_layout()
    plt.show()
    
    land_adjacent_ndwi[nan_vals] = 2
    
    # raster meta
    out_meta.update({"driver": "GTiff",
                      "height": cropped_nir.shape[1],
                      "width": cropped_nir.shape[2],
                      "count": cropped_nir.shape[0],
                      "nodata": 2,
                      "transform": transform})
    
    # save rasters    
    if write:
        morph_name = os.path.join(out_dir, 'morphed.tif')
        with rasterio.open(morph_name, "w", **out_meta) as dest:
            dest.write(morphed_land, 1)
        
        dest = None
        
        ndwi_name = os.path.join(out_dir, 'ndwi.tif')
        with rasterio.open(ndwi_name, "w", **out_meta) as dest:
            dest.write(cropped, 1)
        
        dest = None
        
        print(f'Wrote: {ndwi_name}')
        
        water_name = os.path.join(out_dir, 'land_adjacent.tif')
        with rasterio.open(water_name, "w", **out_meta) as dest:
            dest.write(land_adjacent_ndwi, 1)
        
        dest = None
        
        percentile10_name = os.path.join(out_dir, 'percentile10.tif')
        with rasterio.open(percentile10_name, "w", **out_meta) as dest:
            dest.write(percentile10, 1)
        
        dest = None
        
        print(f'Wrote: {percentile10_name}')
        
        percentile90_name = os.path.join(out_dir, 'percentile90.tif')
        with rasterio.open(percentile90_name, "w", **out_meta) as dest:
            dest.write(percentile90, 1)
        
        dest = None
        
        print(f'Wrote: {percentile90_name}')
    
    return land_adjacent_ndwi, training_data


# %% - training

# plots the learning curve -- relationship between prediction accuracy and data size
def plotLearningCurve(train_size_abs, train_mean, train_std, test_mean, test_std, curve_title):
    plt.plot(train_size_abs, train_mean, color='forestgreen', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_size_abs, train_mean + train_std, train_mean - train_std, alpha=0.3, color='forestgreen')
    plt.plot(train_size_abs, test_mean, color='royalblue', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
    plt.fill_between(train_size_abs, test_mean + test_std, test_mean - test_std, alpha=0.3, color='royalblue')
    plt.title(curve_title)
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy (f1-score)')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    
    return None

# computes and plots learning curve
def compute_learning_curve(clf, x_train, y_train):

    # start_time = time.time() # start time for process timing
    cv = StratifiedKFold(n_splits=5)
    print(f'\nComputing learning curve for {clf}.')
    
    train_size_abs, train_scores, test_scores = learning_curve(
    clf, x_train, y_train, cv=cv, scoring='f1_macro', 
    train_sizes=np.linspace(0.1, 1., 10), random_state=42)
        
    # Calculate training and test mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot the learning curve
    print(f'--Plotting learning curve for {clf}.')
    plotLearningCurve(train_size_abs, train_mean, train_std, test_mean, test_std, curve_title=clf)
    print(f'Test accuracy:\n{test_mean}')
    print(f'Test standard deviation:\n{test_std}')
        
    return None

def subsample(y, X, X_small_classes, y_small_classes):
    # Identify the class with the largest number of samples
    unique_classes, class_counts = np.unique(y, return_counts=True)
    largest_class = unique_classes[np.argmax(class_counts)]
    
    # Create a mask to identify samples belonging to the largest class
    largest_class_mask = (y == largest_class)
    
    # Create a subset of the largest class by randomly sampling a portion of it
    subset_size = int(0.8 * np.sum(largest_class_mask))  # Adjust the portion as needed
    subset_indices = np.random.choice(np.where(largest_class_mask)[0], size=subset_size, replace=False)
    X_largest_class_subset = X[subset_indices]
    y_largest_class_subset = y[subset_indices]
    
    # Combine the subset of the largest class with the smaller classes for training
    X_train = np.vstack((X_largest_class_subset, X_small_classes))
    y_train = np.concatenate((y_largest_class_subset, y_small_classes))

def train_model(water_vals, training_data):
    labels = np.where((water_vals != 0) & (water_vals != 2), 1, water_vals)
    
    water_vals_1d = training_data
    labels_1d = labels.flatten()
    
    print(f'\nTraining else values: {np.count_nonzero(labels_1d == 0)}')
    print(f'Water labels: {np.count_nonzero(labels_1d == 1)}')
    print(f'Nan labels: {np.count_nonzero(labels_1d == 2)}')
    
    # water_vals_1d = np.delete(water_vals_1d, np.where(training_data == 2), axis = 0)
    # labels_1d = np.delete(labels_1d, np.where(training_data == 2), axis=0)
    
    # subsample()
    
    print(f'\nTrainData Shape: {water_vals_1d.shape}\nLabels Shape: {labels_1d.shape}')    
    
    X_train, X_test, Y_train, Y_test = train_test_split(water_vals_1d, labels_1d, 
                                                        test_size=0.3, random_state=40, stratify=labels_1d)
    
    scaler = MinMaxScaler().fit(water_vals_1d)
    # scaler = StandardScaler().fit(water_vals_1d)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'\nX Train Shape: {X_train_scaled.shape}\nY_train Shape: {Y_train.shape}')
    print(f'Water labels: {np.count_nonzero(Y_train == 1)}\n')
    
    clf = RandomForestClassifier(random_state=42, n_jobs=4, n_estimators=40)
    # clf = HistGradientBoostingClassifier(random_state=42, max_iter=500, learning_rate=0.1, max_depth=5)
    # clf = MLPClassifier(random_state=42, max_iter=300, hidden_layer_sizes=(30,30,30))
    # clf = svm.SVC(C=1.0, class_weight='balanced', random_state=42)
    
    # X_learn_scaled = scaler.transform(water_vals_1d)
    # compute_learning_curve(clf, X_learn_scaled, labels_1d)
    
    print(f'Training {clf}')
    model = clf.fit(X_train_scaled, Y_train)
    
    feature_list = ['blue', 'green', 'red', '704', 'nir', 'ndwi', 'pSDBr']
    
    feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False).round(3)
    print(f'\nFeature Importance:\n{feature_importance}\n')
    
    print('--Computing Precision, Recall, F1-Score...')
    classification = classification_report(Y_test, model.predict(X_test_scaled), labels=model.classes_)
    print(f'--Classification Report:\n{classification}')
    
    return water_vals_1d, labels_1d, model

def save_model(model_dir, model_name, model):
    model_name = os.path.join(model_dir, model_name)
    pickle.dump(model, open(model_name, 'wb')) # save the trained Random Forest model
    
    print(f'Saved model: {model_name}')
    
    return None

# %% - prediction

def predict(test_blue, test_green, test_red, test_704, test_nir, shapefile, model):
    # Open the geotiff file
    with rasterio.open(test_green) as green:
        # Read the green band metadata
        prediction_meta = green.meta
        
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_green, transform = mask(green, gdf.geometry, crop=True)
    
    # Open the geotiff file
    with rasterio.open(test_nir) as nir:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_nir, transform = mask(nir, gdf.geometry, crop=True)
        
    # Open the geotiff file
    with rasterio.open(test_704) as b704:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_704, transform = mask(b704, gdf.geometry, crop=True)          
        
    # Open the geotiff file
    with rasterio.open(test_blue) as blue:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_blue, transform = mask(blue, gdf.geometry, crop=True)
            
    # Open the geotiff file
    with rasterio.open(test_red) as red:            
        # Open the shapefile
        gdf = gpd.read_file(shapefile)
        
        # Crop the raster to the shapefile extent
        cropped_red, out_transform = mask(red, gdf.geometry, crop=True)           
    
    # compute ndwi
    ndwi = (cropped_green - cropped_nir) / (cropped_green + cropped_nir)
    
    # compute pSDBr
    pSDBr = np.log(cropped_blue * 1000) / np.log(cropped_red * 1000)  
    pSDBg = np.log(cropped_blue * 1000) / np.log(cropped_green * 1000)  
    
    # shape prediction data
    test_data = np.vstack((cropped_blue.flatten(), 
                               cropped_green.flatten(), 
                               cropped_red.flatten(),
                               cropped_704.flatten(),
                               cropped_nir.flatten(), 
                               ndwi.flatten(), 
                               # pSDBg.flatten(),
                               pSDBr.flatten())).transpose()
    
    scaler = MinMaxScaler().fit(test_data)
    # scaler = StandardScaler().fit(test_data)
    scaled = scaler.transform(test_data)
    scaled[np.isnan(scaled)] = 2
        
    prediction = model.predict(scaled)
    prediction_shape = cropped_red.shape
    
    print(f'\nPrediction (0) values: {np.count_nonzero(prediction == 0)}')
    print(f'Prediction (1) values: {np.count_nonzero(prediction == 1)}')
    
    return prediction_shape, prediction, prediction_meta, pSDBr, out_transform

def plot_prediction(prediction, prediction_shape, pSDBr):
    # reshape
    img = np.reshape(prediction, prediction_shape)
    img = np.moveaxis(img, 0, -1)[:,:,0]
    
    # pSDBr = np.moveaxis(pSDBr, 0, -1)[:,:,0]
    # mask = np.ma.masked_where(img != 1, img)
    img = np.where(img == 2, np.nan, img)
    
    fig = plt.figure()
    # plt.imshow(pSDBr, cmap='gray')
    # plt.imshow(mask, cmap='hot', alpha=0.7)
    plt.imshow(img, cmap='viridis')
    plt.title('Prediction')
    plt.colorbar()
    plt.show()
    
    return img.shape

def save_prediction(prediction, pSDBr, prediction_shape, prediction_meta, out_dir, out_transform):
        
    prediction_name = os.path.join(out_dir, '_prediction.tif')
    pSDBr_name = os.path.join(out_dir, '_pSDBr.tif')
    img = np.reshape(prediction, prediction_shape)
    # img = np.ma.masked_where(img == 1, img)
    
    # raster meta
    prediction_meta.update({"driver": "GTiff", 
                            "height": prediction_shape[1],
                            "width": prediction_shape[2],
                            "count": prediction_shape[0],
                            "nodata": 2, 
                            "transform": out_transform})
    
    # save rasters    
    with rasterio.open(prediction_name, "w", **prediction_meta) as dest:
        dest.write(img) # had to specify '1' here for some reason
        dest = None
        
    print(f'\nSaved prediction to: {prediction_name}')
    
    # save rasters    
    with rasterio.open(pSDBr_name, "w", **prediction_meta) as dest:
        dest.write(pSDBr) # had to specify '1' here for some reason
        dest = None
        
    print(f'Saved pSDBr to: {pSDBr_name}')

    return None


# %% - main

if __name__ == '__main__':
    input_blue = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_492.tif"
    input_green = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_560.tif"
    input_red = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_665.tif"    
    input_nir = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_833.tif"
    input_704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_704.tif"
    # shapefile = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet.shp"
    shapefile = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet_FocusedExtent.shp"
    
    out_dir = r"C:\_ZeroShoreline\Out\Hatteras_20230127"
    model_dir = r'C:\_ZeroShoreline\Model'
    model_name = 'RF_BGR7NWpR.pkl'
    
    # input_blue = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_492.tif"
    # input_green = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_559.tif"
    # input_red = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_665.tif"
    # input__704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_704.tif"
    # input_nir = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_833.tif"
    
    
    
    water_vals, training_data = near_land(input_blue, input_green, input_red, 
                                          input_704, input_nir, shapefile, out_dir, write=False)
    # water_vals_1d, labels_1d, model = train_model(water_vals, training_data)
    # save_model(model_dir, model_name, model)
    # prediction_shape, prediction, prediction_meta, pSDBr, out_transform = predict(test_blue, test_green, test_red, test_704, test_nir, shapefile, model)
    # img_shape = plot_prediction(prediction, prediction_shape, pSDBr)
    # save_prediction(prediction, pSDBr, prediction_shape, prediction_meta, out_dir, out_transform)
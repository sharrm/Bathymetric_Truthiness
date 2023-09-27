# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:44:43 2023

@author: matthew.sharr
"""

import fiona
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import rasterio
import rasterio.mask
from scipy import spatial
from scipy import ndimage
from skimage import feature, filters
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import classification_report, jaccard_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import plot_tree
import warnings


# %% - globals and functions

warnings.filterwarnings('ignore')

def subsample(array1, array2, adjustment):   
    # get indices of rows containing 0 and 1
    indices_with_zeros = np.where(array1 == 0)[0]
    indices_with_ones = np.where(array1 == 1)[0]
    
    # randomly select a subset of rows containing 0
    num_rows_to_select = np.count_nonzero(array1 == 1) * adjustment # Adjust as needed
    rng = np.random.default_rng(0)
    selected_indices_zeros = rng.choice(indices_with_zeros, size=num_rows_to_select, replace=False)
    
    # include the randomly selected rows
    selected_labels = np.concatenate((array1[indices_with_ones], array1[selected_indices_zeros]))
    selected_training = np.vstack((array2[indices_with_ones], array2[selected_indices_zeros]))
    
    return selected_labels, selected_training
    
    
# %% - input

# in492 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_704.tif"    
# in833 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet_FocusedExtent.shp"

# input_blue = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_492.tif"
# input_green = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_559.tif"
# input_red = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_665.tif"
# input__704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_704.tif"
# input_nir = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_833.tif"

# in492 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_704.tif"
# in833 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\FL_Zero.shp"
    
in492 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
in560 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_560.tif"
in665 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_665.tif"
in704 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_704.tif"
in833 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_833.tif"
land = r"C:\_ZeroShoreline\Extent\StCroix_Zero.shp"
    
with fiona.open(land, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open(in492) as blue:
    blue_img, out_transform = rasterio.mask.mask(blue, shapes, crop=True)
    out_meta = blue.meta

with rasterio.open(in560) as green:
    green_img, out_transform = rasterio.mask.mask(green, shapes, crop=True)
    out_meta = green.meta
    
with rasterio.open(in665) as red:
    red_img, out_transform = rasterio.mask.mask(red, shapes, crop=True)
    out_meta = red.meta
    
# with rasterio.open(in704) as rededge:
#     rededge_img, out_transform = rasterio.mask.mask(rededge, shapes, crop=True)
#     out_meta = rededge.meta
    
with rasterio.open(in833) as nir:
    nir_img, out_transform = rasterio.mask.mask(nir, shapes, crop=True)
    out_meta = nir.meta
    
ndwi = (green_img - nir_img) / (green_img + nir_img)
pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)

train_arr = np.vstack((ndwi.flatten(),
                        pSDBr.flatten(),
                        nir_img.flatten(),
                        # rededge_img.flatten(),
                        red_img.flatten(), 
                        green_img.flatten(), 
                        blue_img.flatten()
                      )).transpose()

canny_feat = feature.canny(ndwi[0,:,:], sigma=3)
poi = np.where((canny_feat == 1) & (ndwi[0,:,:] > 0.1), 1, 0 )
# mask_land = np.where(ndwi > 0, 0, 1)
# water_vals = np.where(land_dilate != mask_land, 1, 0)
# canny_water = np.where(water_vals == canny_dilate, 1, 0)
# pSDBr_water = np.where(canny_feat, pSDBr, canny_feat)
# pSDBr_nozeros = pSDBr_water[pSDBr_water != 0]
# percentile10 = np.nanpercentile(np.where(poi == 1), 10)
# percentile90 = np.nanpercentile(np.where(poi == 1), 90)
# poi90 = np.where(poi > percentile90, poi, 0)

# plt.imshow(poi)
# plt.colorbar()
# plt.show()

out_canny = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'Wake_poi2.tif')
# out_canny_dilate = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'canny_dilate.tif')
# out_ndwi = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'ndwi.tif')
# out_sobel = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'sobel.tif')
# out_scharr = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'scharr.tif')
# out_laplace = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'laplace.tif')
# out_water = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'water.tif')
# out_pSDBr = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'pSDBr.tif')
# out_pSDBr_water = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'pSDBr90_water.tif')

out_image = canny_feat
out_meta.update({"driver": "GTiff",
                  "height": out_image.shape[0],
                  "width": out_image.shape[1],
                  "transform": out_transform})

# with rasterio.open(out_canny, 'w', **out_meta) as dst:
#     dst.write(poi, 1)   
    
# dst = None

# %% - Train

# get random rows

train_arr = np.nan_to_num(train_arr)
train_labels = poi.flatten()
# train_labels = poi90.flatten()

training_labels, training_selection = subsample(train_labels, train_arr, 20)

scaler = MinMaxScaler().fit(training_selection)
# scaled_water = scaler.transform(train_arr)
# clf = KNeighborsClassifier(2)
# clf = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
# clf = AdaBoostClassifier(random_state=42, learning_rate=1., base_estimator=RandomForestClassifier(n_estimators=100))
# clf = GaussianNB()
clf = RandomForestClassifier(n_jobs=4, n_estimators=20, random_state=42)
# clf = HistGradientBoostingClassifier(random_state=42, max_iter=500, learning_rate=0.2, l2_regularization=0.2)
# clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
# clf = MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(10,10,10,10,10))
# clf = svm.SVC(random_state=42, kernel='rbf', gamma='auto')
# clf.decision_path()
x_train, x_test, y_train, y_test = train_test_split(training_selection, training_labels, test_size=0.33, random_state=42, stratify=training_labels)
scaled_Xtrain = scaler.transform(x_train)
scaled_Xtest = scaler.transform(x_test)

print(f'Training: {clf}\n')
model = clf.fit(scaled_Xtrain, y_train)

print(classification_report(y_test, model.predict(x_test)))
print(f'Jaccard: {jaccard_score(y_test, model.predict(x_test), average="macro"):.3f}')
print(f'Matthews: {matthews_corrcoef(y_test, model.predict(x_test)):.3f}\n')

# print(model.feature_importances_)
feature_list = ['ndwi', 'pSDBr', 'nir', 'red', 'green', 'blue'] # '704',

# plt.figure(figsize=(12, 8))
# plot_tree(model.estimators_[0], feature_names=feature_list, class_names=['0','1'], filled=True, rounded=True)
# plt.show()

# feature_importance = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False).round(3)
# print(f'\nFeature Importance:\n{feature_importance}\n')

# out_model = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Model', 'RF.pkl')
# with open(out_model, 'wb') as f:
    # pickle.dump(model, f)


# %% - Test

# in492 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_704.tif"    
# in833 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230102\NDWI\S2A_MSI_2023_01_02_15_53_20_T18SVE_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet_FocusedExtent.shp"
# land = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet.shp"

in492 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_492.tif"
in560 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_559.tif"
in665 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_665.tif"
in704 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_704.tif"
in833 = r"C:\_ZeroShoreline\Imagery\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_833.tif"
land = r"C:\_ZeroShoreline\Extent\Hatteras_Inlet_FocusedExtent.shp"

# in492 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_704.tif"
# in833 = r"C:\_ZeroShoreline\Imagery\StCroix_20220129\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\StCroix_Zero.shp"

# in492 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_704.tif"
# in833 = r"C:\_ZeroShoreline\Imagery\HalfMoonShoal_20221209\S2A_MSI_2022_12_09_16_16_30_T17RLH_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\Halfmoon_Zero.shp"

# in492 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_704.tif"
# in833 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20230115\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\FL_Zero2.shp"

# in492 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20211201\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20211201\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20211201\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_665.tif"
# # in704 = r
# in833 = r"C:\_ZeroShoreline\Imagery\FL_Keys_20211201\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\FL_Zero2.shp"

# in492 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_704.tif"
# in833 = r"C:\_ZeroShoreline\Imagery\Saipan_20221203\S2A_MSI_2022_12_03_00_52_56_T55PCS_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\Saipan_Zero.shp"

# in492 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_559.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_704.tif"
# in833 = r"C:\_ZeroShoreline\Imagery\Ponce_20221203\S2B_MSI_2022_12_03_15_08_01_T19QGV_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\Ponce_Zero.shp"

# in492 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_560.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_665.tif"
# in704 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_704.tif"
# in833 = r"C:\_ZeroShoreline\Imagery\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\CapeLookout.shp"

# in492 = r"C:\_ZeroShoreline\Imagery\WakeIsland_20221223\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_492.tif"
# in560 = r"C:\_ZeroShoreline\Imagery\WakeIsland_20221223\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_559.tif"
# in665 = r"C:\_ZeroShoreline\Imagery\WakeIsland_20221223\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_665.tif"
# in833 = r"C:\_ZeroShoreline\Imagery\WakeIsland_20221223\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_833.tif"
# land = r"C:\_ZeroShoreline\Extent\WakeIsland_Zero.shp"

with fiona.open(land, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

with rasterio.open(in492) as blue:
    blue_img, out_transform = rasterio.mask.mask(blue, shapes, crop=True)
    out_meta = blue.meta

with rasterio.open(in560) as green:
    green_img, out_transform = rasterio.mask.mask(green, shapes, crop=True)
    out_meta = green.meta
    
with rasterio.open(in665) as red:
    red_img, out_transform = rasterio.mask.mask(red, shapes, crop=True)
    out_meta = red.meta
    
# with rasterio.open(in704) as rededge:
#     rededge_img, out_transform = rasterio.mask.mask(rededge, shapes, crop=True)
#     out_meta = rededge.meta    
    
with rasterio.open(in833) as nir:
    nir_img, out_transform = rasterio.mask.mask(nir, shapes, crop=True)
    out_meta = nir.meta
    
ndwi = (green_img - nir_img) / (green_img + nir_img)
pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)

test_arr = np.vstack((ndwi.flatten(),
                       pSDBr.flatten(),
                        nir_img.flatten(),
                        # rededge_img.flatten(),
                        red_img.flatten(), 
                        green_img.flatten(), 
                        blue_img.flatten()
                      )).transpose()

test_arr = np.nan_to_num(test_arr)
test_scaler = MinMaxScaler().fit(test_arr)
test_scaled = test_scaler.transform(test_arr)

# in_model = r'C:\_ZeroShoreline\Model\RF_BGR7NWpR_StCroix.pkl'

# with open(in_model, 'rb') as f:
#     model = pickle.load(f)

print('Predicting...')

prediction = model.predict(test_scaled).reshape((ndwi[0,:,:].shape))

plt.imshow(ndwi[0,:,:], cmap='Greys')
plt.imshow(prediction, cmap='Reds', vmax=0.5, alpha=0.5)
plt.show()

# percentage for each class
percentage_zeros = (np.count_nonzero(prediction == 0) / prediction.size) * 100
percentage_ones = (np.count_nonzero(prediction == 1) / prediction.size) * 100

# Print the results
print(f"\nQuantity 0: {np.count_nonzero(prediction == 0):,} ({percentage_zeros:.2f}%)")
print(f"Quantity 1: {np.count_nonzero(prediction == 1):,} ({percentage_ones:.2f}%)")



out_image = prediction
out_meta.update({"driver": "GTiff",
                  "height": out_image.shape[0],
                  "width": out_image.shape[1],
                  "transform": out_transform})

out_prediction = os.path.join(r"C:\_ZeroShoreline\Out\FL_20230115",'FL_20230115_prediction.tif')

# with rasterio.open(out_prediction, 'w', **out_meta) as dst:
#     dst.write(out_image, 1)   
    
# dst = None
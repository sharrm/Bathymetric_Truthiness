# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:44:43 2023

@author: matthew.sharr
"""

import fiona
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import rasterio
import rasterio.mask
from scipy import spatial
from scipy import ndimage
from skimage import feature, filters
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# %% - input

in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Training\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_492.tif"
in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Training\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_560.tif"
in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Training\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_665.tif"
in833 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Training\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_833.tif"
land = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\FL_Zero.shp"

in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\WakeIsland\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_492.tif"
in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\WakeIsland\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_559.tif"
in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\WakeIsland\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_665.tif"
in833 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\WakeIsland\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_833.tif"
land = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\WakeIsland_Zero.shp"

# in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\StCroix\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
# in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\StCroix\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_560.tif"
# in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\StCroix\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_665.tif"
# in833 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\StCroix\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_833.tif"
# land = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\StCroix_Zero.shp"

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
    
with rasterio.open(in833) as nir:
    nir_img, out_transform = rasterio.mask.mask(nir, shapes, crop=True)
    out_meta = nir.meta
    
ndwi = (nir_img - green_img) / (nir_img + green_img)
pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)

train_arr = np.vstack((ndwi.flatten(),
                       pSDBr.flatten(),
                       nir_img.flatten(),
                        red_img.flatten(), 
                        green_img.flatten(), 
                       blue_img.flatten()
                      )).transpose()

canny_feat = feature.canny(ndwi[0,:,:], sigma=3)
poi = np.where((canny_feat == 1) & (ndwi[0,:,:] < 0.), 1, 0 )
# mask_land = np.where(ndwi > 0, 0, 1)
# water_vals = np.where(land_dilate != mask_land, 1, 0)
# canny_water = np.where(water_vals == canny_dilate, 1, 0)
# pSDBr_water = np.where(canny_feat, pSDBr, canny_feat)
# pSDBr_nozeros = pSDBr_water[pSDBr_water != 0]
# percentile10 = np.nanpercentile(pSDBr_nozeros, 10)
# percentile90 = np.nanpercentile(pSDBr_nozeros, 90)
# pSDBr90 = np.where(pSDBr_water > percentile90, pSDBr_water, 0)

plt.imshow(poi)
plt.colorbar()
plt.show()

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
# water_index = np.where(train_poi != 1)
# train_water = np.delete(train_arr, water_index, axis=0)
# train_labels = np.delete(train_poi, water_index)

scaler = MinMaxScaler().fit(train_arr)
# scaled_water = scaler.transform(train_arr)
clf = RandomForestClassifier(n_jobs=4, n_estimators=20, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(train_arr, train_labels, test_size=0.33, random_state=42, stratify=train_labels)
scaled_Xtrain = scaler.transform(x_train)
scaled_Xtest = scaler.transform(x_test)

model = clf.fit(scaled_Xtrain, y_train)

print(classification_report(y_test, model.predict(x_test)))
print(model.feature_importances_)

out_model = os.path.join(r'C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Model', 'RF.pkl')
with open(out_model, 'wb') as f:
    pickle.dump(model, f)


# %% - Test

# in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\StCroix\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_492.tif"
# in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\StCroix\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_560.tif"
# in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\StCroix\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_665.tif"
# in833 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\StCroix\S2A_MSI_2022_01_29_14_58_03_T20QKE_L2R_rhos_833.tif"
# land = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\StCroix_Zero.shp"

# in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\WakeIsland\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_492.tif"
# in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\WakeIsland\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_559.tif"
# in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\WakeIsland\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_665.tif"
# in833 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Testing\WakeIsland\S2B_MSI_2022_12_23_23_31_09_T58QFG_L2R_rhos_833.tif"
# land = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\WakeIsland_Zero.shp"

in492 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Training\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_492.tif"
in560 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Training\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_560.tif"
in665 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Training\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_665.tif"
in833 = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Training\S2A_MSI_2023_01_15_16_06_24_T17RNH_L2R_rhos_833.tif"
land = r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Land\FL_Zero.shp"

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
    
with rasterio.open(in833) as nir:
    nir_img, out_transform = rasterio.mask.mask(nir, shapes, crop=True)
    out_meta = nir.meta
    
ndwi = (nir_img - green_img) / (nir_img + green_img)
pSDBr = np.log(blue_img * 1000) / np.log(red_img * 1000)

test_arr = np.vstack((ndwi.flatten(),
                       pSDBr.flatten(),
                        nir_img.flatten(),
                        red_img.flatten(), 
                        green_img.flatten(), 
                        blue_img.flatten()
                      )).transpose()

test_arr = np.nan_to_num(test_arr)
test_scaler = MinMaxScaler().fit(test_arr)
test_scaled = test_scaler.transform(test_arr)

prediction = model.predict(test_scaled).reshape((ndwi[0,:,:].shape))

plt.imshow(ndwi[0,:,:], cmap='Greys')
plt.imshow(prediction, cmap='Reds', vmax=0.5, alpha=0.5)
plt.show()

# unique, counts = np.unique(prediction, return_counts=True)
# print(unique, counts)

# with open(out_model, 'rb') as f:
#     model = pickle.load(f)

# plt.imshow(canny_feat, cmap='Reds', vmax=0.5, alpha=0.4)
# plt.show()

out_image = prediction
out_meta.update({"driver": "GTiff",
                  "height": out_image.shape[0],
                  "width": out_image.shape[1],
                  "transform": out_transform})

out_prediction = os.path.join(r"C:\Users\Matthew.Sharr\Documents\NGS\ZeroShorline\Out",'prediction.tif')

# with rasterio.open(out_prediction, 'w', **out_meta) as dst:
#     dst.write(out_image, 1)   
    
# dst = None
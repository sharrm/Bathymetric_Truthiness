# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:58:35 2022

@author: sharrm

Scipt to begin digging into generalizing ML model
"""

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy
import os
import sys
import cv2

# %% -- input data

#train
keylargo_raster = r"P:\Thesis\Training\KeyLargo\_Train\_Bands_11Band\_Composite\KeyLargo_composite.tif"
puertoreal_raster = r"P:\Thesis\Training\PuertoReal\_Train\_Bands_11Band\_Composite\PuertoReal_composite.tif"
portland_raster = r"P:\Thesis\Training\Portland\_Bands_11Band\_Composite\Portland_composite.tif"

#test
oldorchard_raster = r"P:\Thesis\Test Data\OldOrchard\_Bands_11Band\_Composite\OldOrchard_composite.tif"
rockyharbor_raster = r"P:\Thesis\Test Data\RockyHarbor\_Bands_11Band\_Composite\RockyHarbor_composite.tif"
greatlakes_raster = r"P:\Thesis\Test Data\GreatLakes\_Bands_11Band\_Composite\GreatLakes_composite.tif"

rasters = [keylargo_raster, puertoreal_raster, portland_raster, rockyharbor_raster, greatlakes_raster]

# %% - functions

scale = MinMaxScaler()

# def scaleData(data):
#     # print(np.min(data), np.max(data))
#     return (data - np.min(data)) / (np.max(data) - np.min(data))

# %% -- read data into arrays

band1 = gdal.Open(rasters[0])

bandcount = band1.RasterCount # number of raster bands

result = []

for r in rasters:
    band = gdal.Open(r)
    if band.RasterCount != bandcount:
        print('Unequal number of bands in input rasters')
            
    for i in range(1, bandcount + 1):
        # read rasters as arrays
        b1 = band.GetRasterBand(i).ReadAsArray()
        
        # remove infinity, convert zeros to nans, remove outliers, scale data
        b1[b1 == -9999.] = 0.
        b1[b1 == np.nan] = 0.
        b1[(b1 < np.mean(b1) - 3 * np.std(b1)) | (b1 > np.mean(b1) + 3 * np.std(b1))] = 0
        b1 = np.nan_to_num(b1)
        # b1 = b1[b1 != 0]
    
        result.append(scale.fit_transform(b1))
    
# convert to arrays
# results = np.vstack(result)


# %% - organize histograms &  kl divergence

bins = 100
hist = np.zeros((bins, bandcount*len(rasters)))
# kl_div = np.zeros((bins, bandcount*len(rasters)))

for i in range(0, len(result)):
    hist[:,i], bins1 = np.histogram(result[i], bins=bins)

# for i in range(0, bandcount):
#     kl_div[:,i] = scipy.special.kl_div(hist[i], hist[i+bandcount])


# %% - plotting

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

col = 0
cmaps = []
for i in range(0, len(rasters)):
    cmaps.append((np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)))

# fig1, ax1 = plt.subplots(bandcount,1,figsize=(9,8))

for i in range(0,bandcount): # for each band
    for j in range(0, len(rasters)): # each raster
        plt.stairs(hist[1:,col], alpha = 0.5, fill=True, label=f'{os.path.basename(rasters[j])}', color=cmaps[j])
        col += 1
        print(i, col)

        # right now the issue is plotting the incorrect distribution
        # need to increment column in the hist function on line 107
        # col += j + i * bandcount
        # print(i, col)

    plt.legend(loc='upper right', prop={'size': 7})
    plt.title(f'{feature_list[i]}', y=1.0, pad=-14, fontsize=11)
    plt.xlabel('Bin')
    plt.ylabel('Count')
    plt.show()


    #     ax1[i].stairs(hist[:,col], alpha = 0.6, fill=True, label=f'Raster {j}; Band {i+1}', color=cmaps[j])
    #     ax1[i].legend(loc='upper right')
    #     ax1[i].set_title(f'{feature_list[i]}', y=1.0, pad=-14)
    #     ax1[i].set_xlabel('Bin')
    #     ax1[i].set_ylabel('Count')
    #     col+=1

    # fig1.suptitle('Distribution of Image Band Values')
    # fig1.tight_layout()
    # plt.show()

# fig2, ax2 = plt.subplots(bandcount,1,figsize=(9,8))

# for i in range(0,bandcount*len(rasters)):
#     ax2[i].bar(np.arange(bins) + 0.3, kl_div[:,i], width = 0.5, alpha = 0.6, color=(0.9, 0.5, 0.1))
#     ax2[i].set_title(f'{feature_list[i]}', y=1.0, pad=-14)
#     ax2[i].set_xlabel('Bin')
#     ax2[i].set_ylabel('KL Divergence')

# fig2.suptitle('KL Divergence')
# fig2.tight_layout()
# plt.show()


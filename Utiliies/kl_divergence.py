# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:58:35 2022

@author: sharrm

Scipt to begin digging into generalizing ML model
"""

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import sys
import cv2

# %% -- input data

raster1 = r"C:\Users\sharrm\Box\CE 560 - Final Project\Test Data\KeyLargo_Composite_4_compressionless.tif"
raster2 = r"C:\Users\sharrm\Box\CE 560 - Final Project\Test Data\KeyLargoSW\KeyLargoSW_Composite4.tif"
# raster1 = r"C:\Users\sharrm\Box\CE 560 - Final Project\Test Data\KeyLargo_Composite_10_compressionless.tif"
# raster2 = r"C:\Users\sharrm\Box\CE 560 - Final Project\Test Data\KeyLargoSW\KeyLSW_Composite_10.tif"

# %% - functions

def scaleData(data):
    # print(np.min(data), np.max(data))
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# %% -- read data into arrays

band1 = gdal.Open(raster1)
band2 = gdal.Open(raster2)

if band1.RasterCount != band2.RasterCount:
    print('Unequal number of bands in input rasters')

bandcount = band1.RasterCount # number of raster bands

bands1 = []
bands2 = []

for i in range(1, bandcount + 1):
    # read rasters as arrays
    b1 = band1.GetRasterBand(i).ReadAsArray()
    b2 = band2.GetRasterBand(i).ReadAsArray()
    
    # remove infinity, convert zeros to nans, remove outliers, scale data
    b1 = np.nan_to_num(b1)
    b2 = np.nan_to_num(b2)
    b1[b1 == -9999] = 0
    b2[b2 == -9999] = 0
    b1 = b1[b1 != 0]
    b2 = b2[b2 != 0]
    b1[(b1 < np.mean(b1) - 3 * np.std(b1)) | (b1 > np.mean(b1) + 3 * np.std(b1))] = 0
    b2[(b2 < np.mean(b2) - 3 * np.std(b2)) | (b2 > np.mean(b2) + 3 * np.std(b2))] = 0
    b1 = b1[b1 != 0]
    b2 = b2[b2 != 0]

    b1 = scaleData(b1)
    b2 = scaleData(b2)
    
    # append to lists
    bands1.append(b1)
    bands2.append(b2)
    
# convert to arrays
bands1 = np.array(bands1)
bands2 = np.array(bands2) # .transpose((1,2,0))

# %% - organize histograms

bins = 100
hist1 = np.zeros((bins, bandcount))
hist2 = np.zeros((bins, bandcount))

for i in range(0,bandcount):
    hist1[:,i], bins1 = np.histogram(bands1[i], bins=bins)
    hist2[:,i], bins2 = np.histogram(bands2[i], bins=bins)

# %% - kl divergence

kl_div = scipy.special.kl_div(hist1, hist2)
# kl_div[kl_div == np.inf] = np.nan

# %% - plotting
fig, ax = plt.subplots(bandcount,1,figsize=(9,8))

for i in range(0,bandcount):
    ax[i].stairs(hist1[:,i], alpha = 0.6, fill=True, label=f'Raster A: Band {i+1}', color=(0.1, 0.5, 0.8))
    ax[i].stairs(hist2[:,i], alpha = 0.6, fill=True, label=f'Raster B: Band {i+1}', color=(0.3, 0.7, 0.4))
    # ax[i].bar(np.arange(bins) + 0.3, kl_div[:,i], width = 0.5, alpha = 0.6, color=(0.9, 0.5, 0.1), fill=True, label=f'KL Divergence: Band {i+1}')
    ax[i].legend(loc='upper right')
    ax[i].set_title(f'Raster A vs B for Band {i+1}', y=1.0, pad=-14)
    ax[i].set_xlabel('Bin')
    ax[i].set_ylabel('Count')

fig.suptitle('Distribution of Image Band Values')
fig.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(bandcount,1,figsize=(9,8))

for i in range(0,bandcount):
    ax2[i].bar(np.arange(bins) + 0.3, kl_div[:,i], width = 0.5, alpha = 0.6, color=(0.9, 0.5, 0.1))
    ax2[i].set_title(f'KL Divergence for Band {i+1}', y=1.0, pad=-14)
    ax2[i].set_xlabel('Bin')
    ax2[i].set_ylabel('KL Divergence')

fig2.suptitle('KL Divergence')
fig2.tight_layout()
plt.show()


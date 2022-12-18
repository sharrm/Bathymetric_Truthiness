# -*- coding: utf-8 -*-
"""
@author: sharrm
"""

# import sys
# import matplotlib.pyplot as plt
# import rasterio.mask
# # from rasterio import plot
# # from rasterio.plot import show
# import fiona
# import geopandas as gpd
# # from osgeo import gdal
# import linear_regression as slr

import os
import numpy as np
import rasterio

band1 = r"U:\ce567\sharrm\Lab7\dataset_files\psdbred.tif"
band2 = r"U:\ce567\sharrm\Lab7\dataset_files\psdbgreen.tif"

# open the rasters
red = rasterio.open(band1)
green = rasterio.open(band2)
nodata = red.nodata # current metadata nodata value from input data
out_meta = red.meta # current metadata
out_transform = red.transform # current input data transform information

# read the rasters to arrays
red = red.read(1)
green = green.read(1)

# linear regression
SDBred = (-203.18 * red + 194.2) * (-1)
SDBgreen = (-203.18 * green + 194.2) * (-1)

# Caballero/Stumpf switching
# areas shoaler than 2m = SDBred, everything else is SDBgreen
deep = np.where((SDBred > 2) & (SDBgreen > 3.5), SDBgreen, SDBred)

# Caballero/Stumpf weighting
alpha = (3.5 - SDBred) / (3.5 - 2)
beta = 1 - alpha
weighted = ((alpha * SDBred) + (beta * SDBgreen)) * (-1)

# final switched array with red < 2m, weighted between 2-3.5m, and green > 3.5m
switched = np.where((SDBred > 2) & (SDBgreen < 3.5), weighted, deep)
switched[np.isinf(switched)] = nodata
switched[np.isnan(switched)] = nodata

# write switched raster
outraster_name = os.path.join(os.path.dirname(band1), 'switched_SDB.tif')

# update spatial information
out_meta.update({"driver": "GTiff",
                  "height": switched.shape[0],
                  "width": switched.shape[1],
                  "transform": out_transform})

with rasterio.open(outraster_name, "w", **out_meta) as dest:
    dest.write(switched, 1) # write array to raster

# trying to plot the source of the pixel value
red_source = np.where((SDBred < 2) & (np.isfinite(SDBred)), 0, SDBgreen)
green_source = np.where((SDBred > 2) & (SDBgreen > 3.5), 4, red_source)
data_source = np.where((SDBred >= 2) & (SDBgreen <= 3.5) & (np.isfinite(SDBred)), 2, green_source)
data_source[np.isinf(data_source)] = nodata
    
source = os.path.join(os.path.dirname(band1), 'SDBsource.tif')
  
# with rasterio.open(source, 'w', **out_meta) as dest:
#     dest.write(data_source, 1)

    # had to specify '1' here for some reason

# this will make a composite
# remember to update count, and shift the np depth axis to the beginning

# method for creating composite
# comp = np.dstack((red, green))
# comp = np.moveaxis(comp.squeeze(),-1,0)

# print(f'comp shape {comp.shape}')
#
# out_meta.update({"driver": "GTiff",
#                   "height": comp.shape[1],
#                   "width": comp.shape[2],
#                   "count": comp.shape[0],
#                   'compress': 'lzw',
#                   "transform": out_transform})
#
# print(comp.shape)
# print(out_meta)
#
# with rasterio.open(os.path.join(os.path.dirname(band1), 'comp.tif'), "w", **out_meta) as dest:
#     dest.write(comp) # had to specify '1' here for some reason


# input files
# blueband = r"P:\SDB\Florida Keys\Popcorn\Test_Files\masked_492.tif"
# greenband = r"P:\SDB\Florida Keys\Popcorn\Test_Files\masked_560.tif"
# redband = r"P:\SDB\Florida Keys\Popcorn\Test_Files\masked_665.tif"
# mask = r"U:\ce567\sharrm\Lab7\dataset_files\mask_buck.shp"

# def read_band(band):
#     with rasterio.open(masked_rasters['blue']) as src:
#         read_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
#         out_meta = src.meta
       
#     return read_image, out_meta, out_transform

# def relative_bathymetry(band1, band2):
#     band1, ref, transform = read_band(band1)
#     band2, ref, transform = read_band(band2)

#     # Stumpf algorithm
#     ratiologs = np.log(1000 * band1) / np.log(1000 * band2)  
    
#     return ratiologs, ref, transform

# def write_raster(band1, band2):
#     output_rol = relative_bathymetry(band1, band2)
    
#     # output raster filename with path
#     outraster_name = os.path.join(os.path.dirname(band1), 'ratio_of_logs.tif')
    
#     # write ratio between bands to a file
#     with rasterio.open(outraster_name, "w", **out_meta) as dest:
#         dest.write(ratioImage)
    
#     # close the file
#     dest = None
        
#     return None

# write_raster(blueband, redband)

# test arrays
# red = np.array([[1, 1, 1, 1, 1],
#                 [1.5, 1.5, 1.5, 1.5, 1.5],
#                 [1.5, 1.5, 1.5, 1.5, 1.5], 
#                 [2.5, 2.5, 2.5, 2.5, 2.5],
#                 [3, 3, 3, 3, 3],
#                 [3, 3, 3, 3, 3],
#                 [4, 4, 4, 4, 4]])

# green = np.array([[3, 3, 4, 4, 4], 
#              [5, 5, 5, 5, 5],
#              [5, 5, 5, 5, 5],
#              [3, 3, 3, 3, 3],
#              [7, 7, 7, 7, 7],
#              [7, 7, 7, 7, 7],
#              [9, 9, 9, 9, 9]])









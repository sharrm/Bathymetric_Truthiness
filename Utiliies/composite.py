# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:54:10 2023

@author: sharrm
"""

import sys
import os
import numpy as np
# import matplotlib.pyplot as plt
import rasterio
# import rasterio.mask
# import richdem as rd
from osgeo import gdal
# from scipy.ndimage.filters import uniform_filter
# from skimage import feature, filters
# from rasterio import plot
# from rasterio.plot import show
import fiona
#import geopandas as gpd
# from osgeo import gdal

directory = r'P:\Thesis\Training\PuertoReal\_Predictors\Bands'
# directory = r'P:\Thesis\Test Data\OldOrchard'
output_dir = directory + '\_Composite'
output_name = 'puerto_real_composite.tif'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def composite(directory, output_name):
    
    output_composite = os.path.join(output_dir, output_name)
    
    dir_list = [file for file in os.listdir(directory) if file.endswith(".tif")]
    
    bands = []
    
    need_meta_trans = True
    meta_transform = []
    
    for file in dir_list:
        band = rasterio.open(os.path.join(directory, file))
        
        print(f'Read {file}')
        
        if need_meta_trans:
            out_meta = band.meta
            out_transform = band.transform  
            meta_transform.append(out_meta)
            meta_transform.append(out_transform)
            need_meta_trans = False
        
        bands.append(band.read(1))        
        band = None
        
    # remember to update count, and shift the np depth axis to the beginning
    # method for creating composite
    comp = np.dstack(bands)
    comp = np.rollaxis(comp, axis=2)
    
    out_meta, out_transform = meta_transform
    
    out_meta.update({"driver": "GTiff",
                      "height": comp.shape[1],
                      "width": comp.shape[2],
                      "count": comp.shape[0],
                      # 'compress': 'lzw',
                      "transform": out_transform})
    
    # print(comp.shape)
    # print(out_meta['count'])
    print(f'\nWriting compsite image...\n')
    
    with rasterio.open(output_composite, "w", **out_meta) as dest:
        dest.write(comp) # had to specify '1' here for some reason

    dest = None
    
    print(f'Output {comp.shape[0]} band composite image here:\n{output_composite}')
        
    return output_composite, comp, meta_transform

composite_name, comp, meta_transform = composite(directory, output_name)

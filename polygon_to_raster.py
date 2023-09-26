# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:43:02 2023

@author: sharrm
"""

import geopandas as gpd
import matplotlib.pyplot as plt
# from pytictoc import TicToc
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.enums import MergeAlg
from rasterio.plot import show
import numpy as np
import os

# t = TicToc()
# t.tic()

poly_to_raster = True

# 
# "P:\_RSD\Data\Imagery\_turbidTraining_rhos\Hatteras_20230102\_Features_13Bands\_Composite\Hatteras_Ex20230102_13Bands_composite_20230825_1158.tif"
# 'P:\\_RSD\\Data\\Imagery\\_turbidTraining_rhos\\Lookout_20230306\\_Features_13Bands\\_Composite\\Lookout_Ex20230306_13Bands_composite_20230825_1130.tif'

polygon = r"C:\_Turbidity\Extents\_turbidTrainingExMC\Maine_20221006Ex4C.shp"
composite_raster = r'C:\\_Turbidity\\Imagery\\_turbidTraining_rhos\\Maine\\_Features_8Bands\\_Composite\\Maine_20221006Ex4C_9Bands_composite_20230914_0845.tif'

# binary_raster = r"P:\Thesis\Samples\Raster" + '\\' + os.path.basename(polygon).split('.')[0] + '_TF.tif'
binary_raster = r"C:\_Turbidity\Masks\_turbidTrainingMasks" + '\\' + os.path.basename(polygon).split('.')[0] + '_TF.tif'

if poly_to_raster:   
    # Load polygon
    vector = gpd.read_file(polygon)
    raster = rasterio.open(composite_raster)
    out_transform = raster.transform 
    out_meta = raster.meta
    out_shape = raster.shape
    
    geom = []
    for shape in range(0, len(vector['geometry'])):
        geom.append((vector['geometry'][shape], vector['Truthiness'][shape]))
    
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html
    rasterized = features.rasterize(geom,
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        all_touched=False,
        default_value=1,
        dtype=None)
    
    # Plot raster
    fig, ax = plt.subplots(1, figsize = (10, 10))
    show(rasterized, ax = ax)

    # raster_name = os.path.basename(polygon).replace('shp', 'tif')
    # output_raster = os.path.join(os.path.abspath(os.path.join(os.path.dirname(polygon),
    #                                                           '..', 'Raster')), raster_name)
    
    out_meta.update({"driver": "GTiff",
                      "height": out_shape[0],
                      "width": out_shape[1],
                      "count": 1,
                      "nodata": 0.,
                      "transform": out_transform})
    
    with rasterio.open(binary_raster, 'w', **out_meta) as dest:
        dest.write(rasterized, 1)
        
    dest = None
    
    # print(f"\nWrote {output_raster}")
    print(f"\nWrote {binary_raster}")
    
# t.toc()
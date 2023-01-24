# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:43:02 2023

@author: sharrm
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.enums import MergeAlg
from rasterio.plot import show
import numpy as np
import os

# # 11:09:47 From  Michael Olsen  to  Everyone:
# # 	https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm
# # 11:10:45 From  Michael Olsen  to  Everyone:
# # 	https://www.esri.com/training/catalog/5eb18cf2a7a78b65b7e26134/deep-learning-using-arcgis/

polygon = r"P:\Thesis\Samples\Polygon\KeyLargoExtent.shp"
poly_to_raster = True

if poly_to_raster:   
    # Load polygon
    vector = gpd.read_file(polygon)
    raster = rasterio.open(r"P:\Thesis\Training\KeyLargo\_Train\_Bands_11Band\_Composite\KeyLargo_composite.tif")
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

    output_raster = os.path.join(os.path.abspath(os.path.join(os.path.dirname(polygon),
                                                              '..', 'Raster')), 'raster.tif')
    
    out_meta.update({"driver": "GTiff",
                      "height": out_shape[0],
                      "width": out_shape[1],
                      "count": 1,
                      "nodata": 0.,
                      "transform": out_transform})
    
    with rasterio.open(output_raster, 'w', **out_meta) as dst:
        dst.write(rasterized, 1)
    
    
    
    
    
    
    
    
    
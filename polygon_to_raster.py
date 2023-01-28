# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 12:43:02 2023

@author: sharrm
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from pytictoc import TicToc
import rasterio
from rasterio import features
from rasterio.transform import from_bounds
from rasterio.enums import MergeAlg
from rasterio.plot import show
import numpy as np
import os

t = TicToc()
t.tic()

# # 11:09:47 From  Michael Olsen  to  Everyone:
# # 	https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm
# # 11:10:45 From  Michael Olsen  to  Everyone:
# # 	https://www.esri.com/training/catalog/5eb18cf2a7a78b65b7e26134/deep-learning-using-arcgis/

# polygon = r"P:\Thesis\Samples\Polygon\KeyLargoExtent.shp"
# composite_raster = r"P:\Thesis\Training\KeyLargo\_Train\_Bands_11Band\_Composite\KeyLargo_composite.tif"
# polygon = r"P:\Thesis\Samples\Polygon\FLKeys_Training.shp"
# composite_raster = r"P:\Thesis\Training\FLKeys\_Bands_11Band\_Composite\FLKeys_composite.tif"
poly_to_raster = True

# polygon = r"P:\Thesis\_Monday\_Polygons\FLKeys_F_Deep.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\FLKeys_F_Turbid.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\FLKeys_T.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\Halfmoon_F_Deep.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\Halfmoon_F_Turbid.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\Halfmoon_T.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\KeyLargo_F.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\KeyLargo_TF.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\NWHI_F_Deep.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\NWHI_T.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\PR_F_Deep_Clean.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\PR_F_Deep_Noise.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\PR_F_Turbid.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\PR_TF.shp"
# polygon = r"P:\Thesis\_Monday\_Polygons\StCroix_F_Deep.shp"
polygon = r"P:\Thesis\_Monday\_Polygons\StCroix_T.shp"

# composite_raster = r"P:\Thesis\_Monday\X_train\FLKeys_F_Deep_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\FLKeys_F_Turbid_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\FLKeys_T_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\Halfmoon_F_Deep_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\Halfmoon_F_Turbid_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\Halfmoon_T_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\KeyLargo_F_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\KeyLargo_TF_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\NWHI_F_Deep_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\NWHI_T_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\PR_F_Deep_Clean_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\PR_F_Deep_Noise_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\PR_F_Turbid_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\PR_TF_composite.tif"
# composite_raster = r"P:\Thesis\_Monday\X_train\StCroix_F_Deep_composite.tif"
composite_raster = r"P:\Thesis\_Monday\X_train\StCroix_T_composite.tif"

binary_raster = r"P:\Thesis\_Monday\Y_train" + '\\' + os.path.basename(polygon).split('.')[0] + '_truthiness.tif'

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
    
t.toc()
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:19:22 2023

@author: sharrm
"""

import numpy as np
import os
import rasterio

rasters = [r"P:\Thesis\Samples\Raster\FLKeys_Training.tif",
            r"P:\Thesis\Samples\Raster\StCroix_Extents_TF_Training.tif",
            r"P:\Thesis\Samples\Raster\FLKeys_Extents_DeepVessel_Training.tif",
            r"P:\Thesis\Samples\Raster\Ponce_Obvious_Training.tif",
            r'P:\Thesis\Masks\Saipan_Mask_NoIsland_TF.tif',
            r"P:\Thesis\Masks\Niihua_Mask_TF.tif",
            r"P:\Thesis\Masks\PuertoReal_Mask_TF.tif",
            r"P:\Thesis\Masks\GreatLakes_Mask_NoLand_TF.tif"
            ]

for tif in rasters:
    band = rasterio.open(tif).read()
    true_positives = np.count_nonzero(band == 2)
    true_negatives = np.count_nonzero(band == 1)
    no_data_value = np.count_nonzero(band == 0)
    
    print(f'\nLocation: {os.path.basename(tif)}')
    print(f'-Percent True: {true_positives / band.size:1f} ({no_data_value:,} True values)')
    print(f'-Percent False: {true_negatives / band.size:1f} ({true_positives:,} False values)')
    print(f'-Percent No Data: {no_data_value / band.size:1f} ( {true_negatives:,} No Data values)')
    
    band = None
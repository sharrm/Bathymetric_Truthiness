# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:29:34 2023

@author: sharrm
"""

import os
import affine
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT


shp = r"P:\_RSD\Data\ETOPO\etopo_chb.shp"
tif = r"P:\_RSD\Data\ETOPO\ETOPO_2022_v1_30s_N90W180_bed.tif"
# existing_tif = r"C:\_Turbidity\Imagery\_turbidTestingETOPO\Hatteras_20230127\S2B_MSI_2023_01_27_15_53_19_T18SVE_L2R_rhos_492.tif"
existing_tif = r"P:\_RSD\Data\Imagery\_turbidTraining_rhos\Chesapeake_20230316\S2A_MSI_2023_03_16_16_02_50_T18SUG_L2R_rhos_492.tif"
# existing_tif = r"P:\_RSD\Data\Imagery\_turbidTraining_rhos\Lookout_20230306\S2A_MSI_2023_03_06_16_03_31_T18SUD_L2R_rhos_492.tif"
out_etopo = os.path.join(r'P:\_RSD\Data\ETOPO', 'ETOPO_Reprojected_ChesapeakeBay.tif')

with rasterio.open(existing_tif, 'r') as dest:
    dst_crs = dest.crs
    dst_bounds = dest.bounds
    dst_height = dest.height
    dst_width = dest.width

# Output image transform
left, bottom, right, top = dst_bounds
xres = (right - left) / dst_width
yres = (top - bottom) / dst_height
dst_transform = affine.Affine(xres, 0.0, left,
                              0.0, -yres, top)

vrt_options = {
    'resampling': Resampling.cubic,
    'crs': dst_crs,
    'transform': dst_transform,
    'height': dst_height,
    'width': dst_width,
    'nodata': 0
}

with rasterio.open(tif) as src:

    with WarpedVRT(src, **vrt_options) as vrt:

        # At this point 'vrt' is a full dataset with dimensions,
        # CRS, and spatial extent matching 'vrt_options'.

        # Read all data into memory.
        data = vrt.read()

        # Process the dataset in chunks.  Likely not very efficient.
        for _, window in vrt.block_windows():
            data = vrt.read(window=window)

        # Dump the aligned data into a new file.  A VRT representing
        # this transformation can also be produced by switching
        # to the VRT driver.
        directory, name = os.path.split(out_etopo)
        outfile = os.path.join(directory, 'aligned-{}'.format(name))
        rio_shutil.copy(vrt, outfile, driver='GTiff')

print(f'Saved: {outfile}')


# %% - libs

# import fiona
# import geopandas as gpd
# import numpy as np
# import os
# from osgeo import gdal, ogr
# import rasterio
# from rasterio.mask import mask
# from rasterio.features import geometry_mask
# from rasterio.transform import Affine
# from rasterio.warp import calculate_default_transform, reproject, Resampling


# %% - mask image

# with fiona.open(shp, "r") as shapefile:
#     geoms2d = [feature["geometry"] for feature in shapefile]

# with rasterio.open(tif) as src:
#     out_image, out_transform = mask(src, geoms2d, crop=True, nodata=0)
#     out_meta = src.meta.copy()

# out_meta.update({"driver": "GTiff",
#                   "height": out_image.shape[1],
#                   "width": out_image.shape[2],
#                   "transform": out_transform,
#                   "nodata": 0})

# out_image = np.where(out_image == -99999, 0, out_image)

# with rasterio.open(out_etopo3, "w", **out_meta) as dest:
#     dest.write(out_image)

# print(f'Wrote: {out_etopo3}')


# %% - resample

# with rasterio.open(out_etopo3) as src:
#     # calculate resampling factor needed to achieve 10m resolution
#     resampling_factor = 802.3493643909026 / 10

#     # resample the raster
#     data = src.read(
#         out_shape=(
#             src.count,
#             int(src.height * resampling_factor),
#             int(src.width * resampling_factor)
#         ),
#         resampling=Resampling.bilinear
#     )

#     # update the transform for the resampled data
#     transform = rasterio.Affine(
#         src.transform.a / resampling_factor,
#         src.transform.b,
#         src.transform.c,
#         src.transform.d,
#         src.transform.e / resampling_factor,
#         src.transform.f
#     )

#     # write the resampled raster
#     with rasterio.open(out_etopo, 'w', driver='GTiff',
#                         height=data.shape[1], width=data.shape[2],
#                         count=src.count, dtype=data.dtype, crs=src.crs,
#                         transform=transform,
#                         nodata=0) as dst:
#         dst.write(data)

# print(f'Wrote: {out_etopo}')


# %% - reproject

# with rasterio.open(existing_tif) as etopo:
#     out_meta = etopo.meta
#     coord_sys = etopo.crs

# dst_crs = coord_sys

# etopo = None

# with rasterio.open(out_etopo) as src:
#     transform, width, height = rasterio.warp.calculate_default_transform(
#         src.crs, dst_crs, src.width, src.height, *src.bounds)

#     with rasterio.open(out_etopo2, 'w', driver='GTiff',
#                         crs=dst_crs, transform=transform,
#                         width=width, height=height, count=src.count, nodata=0,
#                         dtype=src.dtypes[0]) as dst:
#         for i in range(1, src.count + 1):
#             rasterio.warp.reproject(
#                 source=rasterio.band(src, i),
#                 destination=rasterio.band(dst, i),
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=transform,
#                 dst_crs=dst_crs,
#                 nodata=0,
#                 resampling=rasterio.warp.Resampling.bilinear)
    
# print(f'Wrote: {out_etopo2}')


# https://rasterio.readthedocs.io/en/latest/topics/virtual-warping.html
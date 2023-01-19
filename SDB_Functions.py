# -*- coding: utf-8 -*-
"""
@author: sharrm
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
import richdem as rd
from osgeo import gdal
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage import generic_filter
from scipy.ndimage import sobel, prewitt
# from skimage import feature, filters
# from rasterio import plot
# from rasterio.plot import show
import fiona
#import geopandas as gpd
# from osgeo import gdal


def mask_imagery(red, green, blue, nir, in_shp, output_dir):

    # list of bands
    rasters = [blue, green, red, nir]

    # dict to store output file names
    masked_rasters = {}

    #open bounding shapefile
    with fiona.open(in_shp, 'r') as shapefile:
        shape = [feature['geometry'] for feature in shapefile]

    raster_list = []
    #loop through input rasters
    for band in rasters:
        # read raster, extract spatial information, mask the raster using the input shapefile
        with rasterio.open(band) as src:
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
            out_meta = src.meta
            # nodata = src.nodata

        # writing information
        out_meta.update({"driver": "GTiff",
                         # "dtype": 'float32',
                          "height": out_image.shape[1],
                          "width": out_image.shape[2],
                          "transform": out_transform},)

        # simply customizing the output filenames here -- there's probably a better method
        if '492' in band or 'B2' in band: # blue wavelength (492nm)
            outraster_name = os.path.join(output_dir, 'masked_' + os.path.basename(band)[-7:-4] + '.tif')
            masked_rasters['blue'] = outraster_name
        elif '560' in band or 'B3' in band or '559' in band: # green wavelength (560nm)
            outraster_name = os.path.join(output_dir, 'masked_' + os.path.basename(band)[-7:-4] + '.tif')
            masked_rasters['green'] = outraster_name
        elif '665' in band or 'B4' in band: # red wavelength (665nm)
            outraster_name = os.path.join(output_dir, 'masked_' + os.path.basename(band)[-7:-4] + '.tif')
            masked_rasters['red'] = outraster_name
        elif '833' in band or 'B8' in band: # red wavelength (665nm)
            outraster_name = os.path.join(output_dir, 'masked_' + os.path.basename(band)[-7:-4] + '.tif')
            masked_rasters['nir'] = outraster_name

        # write masked raster to a file
        with rasterio.open(outraster_name, "w", **out_meta) as dest:
            dest.write(out_image)

        raster_list.append(outraster_name)
        # close the file
        dest = None

    return True, raster_list, out_meta


def pSDBn (band1, band2, rol_name, output_dir):

    # read blue band
    with rasterio.open(band1) as band1_src:
        band1_image = band1_src.read(1)

    # read green band
    with rasterio.open(band2) as band2_src:
        band2_image = band2_src.read(1)
        out_meta = band2_src.meta

    # Stumpf et al algorithm (2003)
    ratioArrayOutput = np.log(band1_image * 1000.0) / np.log(band2_image * 1000.0)
    
    # output raster filename with path
    outraster_name = os.path.join(output_dir, rol_name + '.tif')
    
    # writing information  
    ratioArrayOutput[np.isnan(ratioArrayOutput)] = 0.0
    ratioArrayOutput[np.isinf(ratioArrayOutput)] = 0.0
    out_meta.update({"dtype": 'float32', "nodata": 0.0})
    
    # write ratio between bands to a file
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(ratioArrayOutput, 1)

    # close the file
    dest = None

    print(f"The ratio raster file is called: {outraster_name}")

    return True, outraster_name, ratioArrayOutput, out_meta


# %% - surface roughness

# in general, when writing a file use one to specify number of bands.

def slope(pSDB_str, slope_name, output_dir):
    slope_output = os.path.join(output_dir, slope_name)
    gdal.DEMProcessing(slope_output, pSDB_str, 'slope') # writes directly to file
    
    return True, slope_output

def stdev(pSDB_str, window, stdev_name, output_dir, out_meta):
    pSDB_slope = rasterio.open(pSDB_str).read(1)
    pSDB_slope[pSDB_slope == -9999.] = 0. # be careful of no data values
    
    print(f'Computing standard deviation of slope within a {window} window...')
    
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html
    std = generic_filter(pSDB_slope, np.std, size=window)
       
    stdev_output = os.path.join(output_dir, stdev_name)
    
    with rasterio.open(stdev_output, "w", **out_meta) as dest:
        dest.write(std, 1)
    
    pSDB_slope = None
    dest = None
    
    return True, stdev_output
    

# from: https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
def window_stdev(pSDBg_slope, window, stdev_name, output_dir, out_meta):
    pSDBg_slope = rasterio.open(pSDBg_slope).read(1)
    pSDBg_slope[pSDBg_slope == -9999.] = 0.
    
    # recommendation; I simply set any nans to 0 below
    # https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    # r,c = pSDBg_slope.shape
    # pSDBg_slope+=np.random.rand(r,c)*1e-6
    
    c1 = uniform_filter(pSDBg_slope, window, mode='reflect')
    c2 = uniform_filter(pSDBg_slope*pSDBg_slope, window, mode='reflect')
    std = np.sqrt(c2 - c1*c1)
    std[std == np.nan] = 0.
    
    stdev_output = os.path.join(output_dir, stdev_name)

    # write stdev slope edges to file
    with rasterio.open(stdev_output, "w", **out_meta) as dest:
        dest.write(std, 1)
    
    return True, stdev_output

# Zevenbergen, L.W., Thorne, C.R., 1987. Quantitative analysis of land surface topography. Earth surface processes and landforms 12, 47–56.
def curvature(pSDB, curvature_name, output_dir, out_meta):
    curvature_output = os.path.join(output_dir, curvature_name)
    
    # sys.stdout = open(os.devnull, 'w')
    rda = rd.rdarray(pSDB, no_data=0.0)
    curve = rd.TerrainAttribute(rda, attrib='curvature')
    # sys.stdout = sys.__stdout__
    
    curve = np.array(curve)
    curve[curve == -9999] = 0.0
    
    # write curvature to file
    with rasterio.open(curvature_output, "w", **out_meta) as dest:
        dest.write(curve, 1)
    
    return True, curvature_output

def tri(pSDB_str, tri_name, output_dir):
    tri_output = os.path.join(output_dir, tri_name)
    gdal.DEMProcessing(tri_output, pSDB_str, 'TRI') # writes directly to file
    
    return True, tri_output

def tpi(pSDB_str, tpi_name, output_dir):
    tpi_output = os.path.join(output_dir, tpi_name)
    gdal.DEMProcessing(tpi_output, pSDB_str, 'TPI') # writes directly to file
    
    return True, tpi_output

def roughness(pSDB_str, roughness_name, output_dir):
    roughness_output = os.path.join(output_dir, roughness_name)
    gdal.DEMProcessing(roughness_output, pSDB_str, 'Roughness') # writes directly to file
    
    return True, roughness_output

# def canny(pSDB, canny_name, output_dir, out_meta):
#     canny_edges = feature.canny(pSDB, sigma=1.0)
#     canny_output = os.path.join(output_dir, canny_name)
    
#     # write canny edges to file
#     with rasterio.open(canny_output, "w", **out_meta) as dest:
#         dest.write(canny_edges, 1)
    
#     return True, canny_output

def sobel_filt(pSDB, sobel_name, output_dir, out_meta):
    sobel_edges = sobel(pSDB)
    sobel_output = os.path.join(output_dir, sobel_name)
    
    # write sobel edges to file
    with rasterio.open(sobel_output, "w", **out_meta) as dest:
        dest.write(sobel_edges, 1)
    
    return True, sobel_output, sobel_edges

def prewitt_filt(pSDB, prewitt_name, output_dir, out_meta):
    prewitt_edges = prewitt(pSDB)
    prewitt_output = os.path.join(output_dir, prewitt_name)
    
    # write prewitt edges to file
    with rasterio.open(prewitt_output, "w", **out_meta) as dest:
        dest.write(prewitt_edges, 1)
    
    return True, prewitt_output, prewitt_edges


# %% - composite

# in a multi-band raster, make band dimension first dimension in array, and use no band number when writing
def composite(dir_list, output_composite):
    
    bands = []
    
    need_meta_trans = True
    meta_transform = []
    
    for file in dir_list:
        band = rasterio.open(file)
        
        print(f'Merging {file} to composite')
        
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
    
    with rasterio.open(output_composite, "w", **out_meta) as dest:
        dest.write(comp) # had to specify '1' here for some reason

    dest = None
        
    return True, output_composite


# %% -

# def stdev_slope(pSDB, window_size, stdev_name, output_dir, out_meta):
#     # window_size = window_size
#     rows, cols = pSDB.shape
#     total_rows = np.arange(window_size, rows+window_size, 1)
#     total_columns = np.arange(window_size, cols+window_size, 1)
#     stdev_slope = np.zeros(pSDB.shape)
#     pSDB = np.pad(pSDB, window_size, mode='constant', constant_values=0.0)

#     # v = np.lib.stride_tricks.sliding_window_view(pSDB, (window_size, window_size))
#     # stdev_slope = np.array([np.std(b) for b in v])
    
#     for i in total_rows:
#         for j in total_columns:
#             window = pSDB[i - window_size : i + window_size, j - window_size : j + window_size]
#             stdev = np.std(window)
#             stdev_slope[i - window_size, j - window_size] = stdev

#     stdev_output = os.path.join(output_dir, stdev_name)

#     # write stdev slope edges to file
#     with rasterio.open(stdev_output, "w", **out_meta) as dest:
#         dest.write(stdev_slope, 1)
    
#     return True, stdev_output, stdev_slope

    # x = np.array([np.arange(5), np.arange(5) + 5, np.arange(5) + 10, np.arange(5) + 20])
    # print(f'x: {x}')
    # x.shape
    # v = np.lib.stride_tricks.sliding_window_view(x, (3,3)) # all windows
    # v.shape
    # print(f'v: {v}')
    # for b in v: # windows in v
    #     print(b+1)
    # r = np.array([b+2 for b in v])    # perform some operation
    # print(f'r: {r}')
    
    # stdev_slope = np.zeros([rows + window_size, cols + window_size])
    
    # for i in total_rows:
    #     for j in total_columns:
    #         window = pSDB[i-window_size : i+window_size+1, j-window_size : j+window_size+1]
    #         stdev = np.std(window)
    #         stdev_slope[i,j] = stdev


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

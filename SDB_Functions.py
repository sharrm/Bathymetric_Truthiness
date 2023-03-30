# -*- coding: utf-8 -*-
"""
@author: sharrm

Updated: 20Mar2023
"""

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
import rasterio.mask
from rasterio.plot import show
from rasterio.transform import from_bounds
import richdem as rd
from osgeo import gdal
from scipy.ndimage import uniform_filter
# from scipy.ndimage import generic_filter
# from scipy.ndimage import sobel, prewitt, laplace, gaussian_gradient_magnitude
# from scipy.ndimage import convolve
import sys
import geopandas as gpd
# from skimage import feature, filters
# from osgeo import gdal


# %% - check boundary
def check_bounds(rgbnir_dir, shapefile):
    raster = [os.path.join(rgbnir_dir, r) for r in os.listdir(rgbnir_dir) if r.endswith('.tif')]
    raster_bounds = rasterio.open(raster[0]).bounds
    
    for r in raster[1:]:
        if raster_bounds != rasterio.open(r).bounds:
            print(f'Unexpected boundary for {r} in {os.path.dirname(rgbnir_dir)}')
        
    # check if shapefiles point locations are inside the bounds of the raster
    shp_bounds = fiona.open(shapefile, 'r').bounds
    
    # check bounds
    eastings_within = np.logical_and(shp_bounds[0] > raster_bounds[0], # left
                                     shp_bounds[2] < raster_bounds[2]) # right
    northings_within = np.logical_and(shp_bounds[1] > raster_bounds[1], # bottom
                                      shp_bounds[3] < raster_bounds[3]) # top
    
    if np.all([eastings_within, northings_within]):
        print(f'{os.path.basename(shapefile)} within bounds of {os.path.basename(rgbnir_dir)} imagery\n')
        return True
    else:
        return False


# %% - features

def mask_imagery(band, in_shp, masked_raster_name):
    #open bounding shapefile
    with fiona.open(in_shp, 'r') as shapefile:
        shape = [feature['geometry'] for feature in shapefile]

    # read raster, extract spatial information, mask the raster using the input shapefile
    with rasterio.open(band) as src:
        out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
        out_meta = src.meta
        
    # writing information
    out_meta.update({"driver": "GTiff",
                     # "dtype": 'float32',
                      "height": out_image.shape[1],
                      "width": out_image.shape[2],
                      "nodata": 0,
                      "count": 1,
                      "transform": out_transform})

    # write masked raster to a file
    with rasterio.open(masked_raster_name, "w", **out_meta) as dest:
        dest.write(out_image)

    # close the file
    dest = None
    return True, masked_raster_name

def rgb_to_cmyk(red_name, green_name, blue_name, output_dir):
    out_meta = rasterio.open(blue_name).meta
    blue_band = rasterio.open(blue_name).read(1)
    green_band = rasterio.open(green_name).read(1)
    red_band = rasterio.open(red_name).read(1)
    
    RGB_SCALE = 255
    CMYK_SCALE = 100

    # rgb [0,255] -> cmy [0,1]
    # c = 1 - red_int/ RGB_SCALE
    # m = 1 - green_band / RGB_SCALE
    # y = 1 - blue_band / RGB_SCALE
    
    c = 1 - red_band
    m = 1 - green_band
    y = 1 - blue_band

    # extract out k [0, 1]
    min_cmy = np.minimum.reduce([c, m, y])
    ck = (c - min_cmy) / (1 - min_cmy)
    mk = (m - min_cmy) / (1 - min_cmy)
    yk = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy
    
    # rescale to the range [0,CMYK_SCALE]
    cmyk = [ck * CMYK_SCALE, mk * CMYK_SCALE, yk * CMYK_SCALE, k * CMYK_SCALE]
    
    cyan_out = os.path.join(output_dir, 'cyan.tif')
    magenta_out = os.path.join(output_dir, 'magenta.tif')
    yellow_out = os.path.join(output_dir, 'yellow.tif')
    black_out = os.path.join(output_dir, 'black.tif')
    
    color_name = [cyan_out, magenta_out, yellow_out, black_out]
    
    for color_arr, outraster_name in zip(cmyk, color_name):
        
        with rasterio.open(outraster_name, "w", **out_meta) as dest:
            dest.write(color_arr,1)

    return True, *color_name

def odi_1(blue_name, green_name, odi_1_name, output_dir):
    out_meta = rasterio.open(blue_name).meta
    blue_band = rasterio.open(blue_name).read(1)
    green_band = rasterio.open(green_name).read(1)
    
    odi_1_arr = (green_band * green_band) / blue_band
    
    out_meta.update({"driver": "GTiff",
                     # "dtype": 'float32',
                      "height": blue_band.shape[0],
                      "width": blue_band.shape[1],
                      "nodata": 0,
                      "count": 1})
    
    # output raster filename with path
    outraster_name = os.path.join(output_dir, odi_1_name)
    
    # write masked raster to a file
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(odi_1_arr, 1)
    
    dest = None
    return True, outraster_name

def odi_2(blue_name, green_name, odi_2_name, output_dir):
    out_meta = rasterio.open(blue_name).meta
    blue_band = rasterio.open(blue_name).read()
    green_band = rasterio.open(green_name).read()
    
    odi_2_arr = (green_band - blue_band) / (green_band + blue_band)
    
    out_meta.update({"driver": "GTiff",
                     # "dtype": 'float32',
                      "height": blue_band.shape[1],
                      "width": blue_band.shape[2],
                      "nodata": 0,
                      "count": 1})
    
    # output raster filename with path
    outraster_name = os.path.join(output_dir, odi_2_name)
    
    # write masked raster to a file
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(odi_2_arr)
    
    dest = None
    return True, outraster_name

def pSDBn (band1, band2, rol_name, output_dir):

    # read first band
    with rasterio.open(band1) as band1_src:
        band1_image = band1_src.read(1)

    # read second band
    with rasterio.open(band2) as band2_src:
        band2_image = band2_src.read(1)
        out_meta = band2_src.meta

    # Stumpf et al algorithm (2003)
    ratioArrayOutput = np.log(band1_image * 1000.0) / np.log(band2_image * 1000.0)
    
    # output raster filename with path
    outraster_name = os.path.join(output_dir, rol_name)
    
    # writing information  
    ratioArrayOutput[np.isnan(ratioArrayOutput)] = 0.0
    ratioArrayOutput[np.isinf(ratioArrayOutput)] = 0.0
    out_meta.update({"dtype": 'float32', "nodata": 0.0})
    
    # write ratio between bands to a file
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(ratioArrayOutput, 1)

    # close the file
    dest = None

    return True, outraster_name

# in general, when writing a file use one to specify number of bands.
def slope(pSDB_str, slope_name, output_dir):
    slope_output = os.path.join(output_dir, slope_name)
    gdal.DEMProcessing(slope_output, pSDB_str, 'slope') # writes directly to file
    return True, slope_output

# from: https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
def window_stdev(pSDB_slope, window, stdev_name, output_dir):
    out_meta = rasterio.open(pSDB_slope).meta
    pSDB_slope = rasterio.open(pSDB_slope).read(1)
    pSDB_slope[pSDB_slope == -9999.] = 0.
    
    # https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    c1 = uniform_filter(pSDB_slope, window, mode='reflect')
    c2 = uniform_filter(pSDB_slope * pSDB_slope, window, mode='reflect')
    std = np.sqrt(c2 - c1*c1)
    std[std == np.nan] = 0.
    
    stdev_output = os.path.join(output_dir, stdev_name)
    
    out_meta.update({"driver": "GTiff",
                      "height": pSDB_slope.shape[0],
                      "width": std.shape[1],
                      "count": 1,
                      "nodata": 0})

    # write stdev slope edges to file
    with rasterio.open(stdev_output, "w", **out_meta) as dest:
        dest.write(std, 1)
    return True, stdev_output

def roughness(pSDB_str, roughness_name, output_dir):
    roughness_output = os.path.join(output_dir, roughness_name)
    gdal.DEMProcessing(roughness_output, pSDB_str, 'Roughness') # writes directly to file
    return True, roughness_output


# %% - composite

# in a multi-band raster, make band dimension first dimension in array, and use no band number when writing
def composite(dir_list, output_composite_name):
    
    bands = []
    
    need_meta_trans = True
    meta_transform = []
    
    for file in dir_list:
        band = rasterio.open(file)
        
        print(f'--Merging {file} to composite')
        
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
                      "nodata": 0,
                      # 'compress': 'lzw',
                      "transform": out_transform})
    
    with rasterio.open(output_composite_name, "w", **out_meta) as dest:
        dest.write(comp) # had to specify '1' here for some reason

    dest = None
    return True, comp.shape[0]


# %% - labeled polygon to raster

def polygon_to_raster(polygon, composite_raster, binary_raster):
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
    return None


# %% - notes

# def canny(pSDB, canny_name, output_dir, out_meta):
#     canny_edges = feature.canny(pSDB, sigma=1.0)
#     canny_output = os.path.join(output_dir, canny_name)
    
#     # write canny edges to file
#     with rasterio.open(canny_output, "w", **out_meta) as dest:
#         dest.write(canny_edges, 1)
    
#     return True, canny_output

# Zevenbergen, L.W., Thorne, C.R., 1987. Quantitative analysis of land surface topography. Earth surface processes and landforms 12, 47–56.
# def curvature(pSDB, curvature_name, output_dir, out_meta):
#     curvature_output = os.path.join(output_dir, curvature_name)
    
#     # sys.stdout = open(os.devnull, 'w')
#     rda = rd.rdarray(pSDB, no_data=0.0)
#     curve = rd.TerrainAttribute(rda, attrib='curvature')
#     # sys.stdout = sys.__stdout__
    
#     curve = np.array(curve)
#     curve[curve == -9999] = 0.0
    
#     # write curvature to file
#     with rasterio.open(curvature_output, "w", **out_meta) as dest:
#         dest.write(curve, 1)
#     return True, curvature_output

# def tri(pSDB_str, tri_name, output_dir):
#     tri_output = os.path.join(output_dir, tri_name)
#     gdal.DEMProcessing(tri_output, pSDB_str, 'TRI', options=gdal.DEMProcessingOptions(alg='Wilson')) # writes directly to file
#     return True, tri_output

# def tpi(pSDB_str, tpi_name, output_dir):
#     tpi_output = os.path.join(output_dir, tpi_name)
#     gdal.DEMProcessing(tpi_output, pSDB_str, 'TPI') # writes directly to file
#     return True, tpi_output

# def sobel_filt(pSDB, sobel_name, output_dir, out_meta):
    
#     sobel_input = rasterio.open(pSDB).read(1)
    
#     sobel_edges = sobel(sobel_input)
#     sobel_output = os.path.join(output_dir, sobel_name)
    
#     # write sobel edges to file
#     with rasterio.open(sobel_output, "w", **out_meta) as dest:
#         dest.write(sobel_edges, 1)
        
#     sobel_input = None
#     dest = None
#     return True, sobel_output, sobel_edges

# def prewitt_filt(pSDB, prewitt_name, output_dir, out_meta):
#     prewitt_edges = prewitt(pSDB)
#     prewitt_output = os.path.join(output_dir, prewitt_name)
    
#     # write prewitt edges to file
#     with rasterio.open(prewitt_output, "w", **out_meta) as dest:
#         dest.write(prewitt_edges, 1)
#     return True, prewitt_output, prewitt_edges

# def laplace_filt(pSDB, laplace_name, output_dir, out_meta):
    
#     laplace_input = rasterio.open(pSDB).read(1)
    
#     laplace_result = laplace(laplace_input)
#     laplace_output = os.path.join(output_dir, laplace_name)
    
#     # write sobel edges to file
#     with rasterio.open(laplace_output, "w", **out_meta) as dest:
#         dest.write(laplace_result, 1)
        
#     laplace_input = None
#     dest = None
#     return True, laplace_output

# def gaussian_gradient_magnitude_filt(pSDB, gauss_name, output_dir, out_meta):
    
#     gauss_input = rasterio.open(pSDB).read(1)
    
#     gauss_result = gaussian_gradient_magnitude(gauss_input, sigma=1)
#     gauss_output = os.path.join(output_dir, gauss_name)
    
#     # write sobel edges to file
#     with rasterio.open(gauss_output, "w", **out_meta) as dest:
#         dest.write(gauss_result, 1)
        
#     gauss_input = None
#     dest = None
#     return True, gauss_output

# def highpass_filt(pSDB, highpass_name, output_dir, out_meta):
    
#     kernel = np.array([[-1, -1, -1],
#                        [-1,  8, -1],
#                        [-1, -1, -1]])
    
#     highpass_input = rasterio.open(pSDB).read(1)
    
#     highpass_result = convolve(highpass_input, kernel)
#     highpass_output = os.path.join(output_dir, highpass_name)
    
#     # write sobel edges to file
#     with rasterio.open(highpass_output, "w", **out_meta) as dest:
#         dest.write(highpass_result, 1)
        
#     highpass_input = None
#     dest = None
#     return True, highpass_output

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

# def stdev(pSDB_str, window, stdev_name, output_dir, out_meta):
#     pSDB_slope = rasterio.open(pSDB_str).read(1)
#     pSDB_slope[pSDB_slope == -9999.] = 0. # be careful of no data values
    
#     print(f'Computing standard deviation of slope within a {window} window...')
    
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.generic_filter.html
#     std = generic_filter(pSDB_slope, np.std, size=window)
       
#     stdev_output = os.path.join(output_dir, stdev_name)
    
#     with rasterio.open(stdev_output, "w", **out_meta) as dest:
#         dest.write(std, 1)
    
#     pSDB_slope = None
#     dest = None
    
#     return True, stdev_output
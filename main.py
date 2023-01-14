"""
@author: sharrm
"""

# Identify user defined function files
import SDB_Functions as sdb
import os
# import linear_regression as slr

# %% - inputs

# Identify the input files
# maskSHP = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\Final Project\Other\clipper.shp" # in_shp
# blueInput = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\NewFinal\Sentinel2\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_492.tif" # Sentinel-2 band 
# greenInput = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\NewFinal\Sentinel2\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_560.tif" # Sentinel-2 band
# redInput = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\NewFinal\Sentinel2\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_665.tif" # Sentinel-2 band

# maskSHP = r"C:\Users\sharrm\Box\CE 560 - Final Project\Test Data\KeyLargoSW\KeyLargoSW_Extents.shp" # in_shp
# blueInput = r"P:\Thesis\Imagery\Key Largo\Processed\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_492.tif" # Sentinel-2 band
# greenInput = r"P:\Thesis\Imagery\Key Largo\Processed\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_560.tif" # Sentinel-2 band
# redInput = r"P:\Thesis\Imagery\Key Largo\Processed\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_665.tif" # Sentinel-2 band
# nirInput = r"P:\Thesis\Imagery\Key Largo\Processed\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_833.tif"

level1C = True
# level1C = False

# maskSHP = r'P:\Thesis\Extents\Puerto_Real_Extents.shp'
# directory = r'P:\Thesis\Training\PuertoReal\_Predictors'
maskSHP = r"P:\Thesis\Extents\OldOrchard_Extents.shp"
directory = r'P:\Thesis\Test Data\OldOrchard'
output_dir = directory + '\_Bands'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# sort processed bands
if level1C:
    for file in os.listdir(directory):
        if file.endswith('.tif') and 'rhos_492' in file:
            blueInput =  os.path.join(directory, file)
        elif file.endswith('.tif') and 'rhos_560' in file or 'rhos_559' in file:
            greenInput =  os.path.join(directory, file )
        elif file.endswith('.tif') and 'rhos_665' in file:
            redInput =  os.path.join(directory, file)
        elif file.endswith('.tif') and 'rhos_833' in file:
            nirInput =  os.path.join(directory, file)
else:
    pass

# %% - band masking
# returns of list of masked bands for each wavelength
mTF, maskOutput, out_meta = sdb.mask_imagery(redInput, greenInput, blueInput, nirInput, maskSHP, output_dir)

# display masked file output location for each wavelength
if mTF:
    print(f"The masked files are:\n"
          f"{maskOutput[0]}\n"
          f"{maskOutput[1]}\n"
          f"{maskOutput[2]}\n"
          f"{maskOutput[3]}\n")

    maskedBlue = maskOutput[0]
    maskedGreen = maskOutput[1]
    maskedRed = maskOutput[2]
    maskedNIR = maskOutput[3]
else:
    print("No masked files were returned from the file masking function.")

# %% -- ratio of logs between bands / relative bathymetry

# Start with Green SDB (deeper)
gTF, outraster_name, pSDBg, out_meta = sdb.pSDBgreen(maskedBlue, maskedGreen, rol_name='pSDBg', output_dir=output_dir)

# returns boolean and the pSDBg location
if gTF:
    pSDBg_name = outraster_name
else:
    print("No green SDB raster dataset was returned from the pSDBgreen function.")

rTF, outraster_name, pSDBr, out_meta = sdb.pSDBred(maskedBlue, maskedRed, rol_name='pSDBr', output_dir=output_dir)

# returns boolean and the pSDBr location
if rTF:
    pSDBr_name = outraster_name
else:
    print("No green SDB raster dataset was returned from the pSDBgreen function.")

# %% - surface roughness computation

# GDAL sLope -- pSDBg_name is a string
slopeTF, pSDBg_slope = sdb.slope(pSDBg_name, slope_name='pSDBg_slope.tif', output_dir=output_dir)
if slopeTF:
    print(f'\nWrote: {pSDBg_slope}')
else:
    print('\nCreating pSDBg slope failed...')
    
# stdevslopeTF, pSDBg_stdevslope, stdev_slope = sdb.stdev_slope(pSDBg, window_size=7, stdev_name='pSDBg_stdevslope.tif', output_dir=output_dir, out_meta=out_meta)
stdevslopeTF, pSDBg_stdevslope, stdev_slope = sdb.window_stdev(pSDBg, radius=7, stdev_name='pSDBg_stdevslope.tif', output_dir=output_dir, out_meta=out_meta)
if stdevslopeTF:
    print(f'\nWrote: {pSDBg_stdevslope}')
else:
    print('\nCreating pSDBg stdev slope failed...')

# curvature RichDEM
curTF, pSDBg_curve = sdb.curvature(pSDBg, curvature_name='pSDBg_curvature.tif', output_dir=output_dir, out_meta=out_meta)
if curTF:
    print(f'\nWrote: {pSDBg_curve}')
else:
    print('\nCreating pSDBg curvature failed...')

# # GDAL TRI
# triTF, pSDBg_tri = sdb.tri(pSDBg_name, tri_name='pSDBg_tri.tif', output_dir=output_dir)
# if triTF:
#     print(f'\nWrote: {pSDBg_tri}')
    
# # GDAL TPI
# tpiTF, pSDBg_tpi = sdb.tpi(pSDBg_name, tpi_name='pSDBg_tpi.tif', output_dir=output_dir)
# if tpiTF:
#     print(f'\nWrote: {pSDBg_tpi}')
    
# # GDAL Roughness
# roughTF, pSDBg_roughness = sdb.roughness(pSDBg_name, roughness_name='pSDBg_roughness.tif', output_dir=output_dir)
# if roughTF:
#     print(f'\nWrote: {pSDBg_roughness}')
    
# # Skimage canny edge detection
# # cTF, blue_canny = sdb.canny(maskedBlue, canny_name='blue_canny.tif', output_dir=output_dir, out_meta=out_meta)
# canTF, pSDBg_canny = sdb.canny(pSDBg, canny_name='pSDBg_canny.tif', output_dir=output_dir, out_meta=out_meta)
# if canTF:
#     print(f'\nWrote: {pSDBg_canny}')
    
# sobelTF, pSDBg_sobel, sobel_edges = sdb.sobel(pSDBg, sobel_name='pSDBg_sobel.tif', output_dir=output_dir, out_meta=out_meta)
# if sobelTF:
#     print(f'\nWrote: {pSDBg_sobel}')
    
# test
# stdevslopeTF, pSDBg_stdevslope, stdev_slope = sdb.stdev_slope(pSDBg, window_size=5, stdev_name='pSDBg_stdevslope_2.tif', output_dir=output_dir, out_meta=out_meta)
    
# should probably change to write file in a function
# rugosity worth continueing? 

print('\nBuilding compsite image...\n')

compTF, composite = sdb.composite(output_dir, output_name='old_orchard_composite.tif')

if compTF:
    print(f'\nWrote: {pSDBg_curve}')
else:
    print('\nCreating pSDBg curvature failed...')

    
# %% -



# # can modify this to compute the ratio of logs between other bands
# # for shallow water, the blue and red have been utilized in the literature
# red_SDB_output = sdb.pSDBgreen (maskedBlue, maskedRed)
#
# if red_SDB_output[0]:
#     redSDB = red_SDB_output[1]
# else:
#     print("No green SDB raster dataset was returned from the pSDBgreen function.")


##############################
# Step 3 Simple linear regression

# Identify the ICESat-2 reference dataset
# icesat2 = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\Final Project\ICESat2\icesat2_clipped.csv"
# icesat2 = r"P:\SDB\Anegada\processed_ATL03_20200811115251_07200801_005_01_o_o_clipped.csv"

# Starting with the Green band:
# Identify other parameters
# SDBraster = greenSDB
# col = "green"
# loc = "Puerto_Real"

# Run the function to see the relationship between lidar depth and relative bathymetric depths
# (returns b0 and b1 as a tuple)
# greenSLRcoefs = slr.slr(SDBraster, icesat2, col)

# # Red band next:
# # Identify other parameters
# SDBraster = redSDB
# col = "green"
# loc = "Key_Largo_Florida"
#
# # Run the function to see the relationship between lidar depth and relative bathymetric depths
# # (returns b0 and b1 as a tuple)
# greenSLR = slr(SDBraster, icesat2, col)

# # Next the Red Band:
# # Identify other parameters
# SDBraster = redSDB
# col = "red"
# loc = "Key_Largo_Florida"


################################
# Step 4 Apply SLR to relative bath

# Only green functionality is currently modeled
# SDBraster = greenSDB
# col = 'green'
# loc = "Puerto_Real"

# tf, final_raster, true_bath = slr.bathy_from_slr(SDBraster, greenSLRcoefs, col, loc)

# if final_raster[0]:
#     print(f"The final raster is located at: {final_raster}")
# else:
#     print("Something went wrong with creating the lidar-based SDB raster.")
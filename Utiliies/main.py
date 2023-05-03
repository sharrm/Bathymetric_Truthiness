"""
@author: sharrm
"""

# Identify user defined function files
import SDB_Functions as sdb
import os, sys
from pytictoc import TicToc
# import linear_regression as slr

t = TicToc()
t.tic()


# %% - inputs

level1C = True
# level1C = False
create_rbg_composite = False
full_workflow = True
# create_rbg_composite = True
# full_workflow = False


# directory = r'P:\Thesis\Test Data\TinianSaipan'
# directory = r"P:\Thesis\Training\KeyLargo"
directory = r"P:\Thesis\Test Data\OldOrchard"
# directory = r"P:\Thesis\Training\FLKeys"
# directory = r"P:\Thesis\Training\StCroix"
# directory = r"P:\Thesis\Test Data\Puerto Real"
# directory = r"P:\Thesis\Test Data\GreatLakes"
# directory = r'P:\Thesis\Test Data\RockyHarbor'
# directory = r'P:\Thesis\Test Data\TinianSaipan'
# directory = r'P:\Thesis\Test Data\WakeIsland'

# maskSHP = r"P:\Thesis\Extents\A_Samoa_Airport.shp"
maskSHP = r"P:\Thesis\Extents\OldOrchard_Extents.shp"

predictor_dir = directory + '\_8Band'
composite_name = os.path.basename(maskSHP).split('.')[0] + '_composite.tif'

if not os.path.exists(predictor_dir):
    os.makedirs(predictor_dir)


# %% - band masking
rgb = []

# sort processed bands
if level1C:
    for file in os.listdir(directory):
        if file.endswith('.tif') and 'rhos_492' in file:
            print(f'Found blue band: {file}')
            blueInput =  os.path.join(directory, file)
            rgb.append(blueInput)
        elif file.endswith('.tif') and 'rhos_560' in file or 'rhos_559' in file:
            print(f'Found green band: {file}')
            greenInput =  os.path.join(directory, file )
            rgb.append(greenInput)
        elif file.endswith('.tif') and 'rhos_665' in file:
            print(f'Found red band: {file}')
            redInput =  os.path.join(directory, file)
            rgb.append(redInput)
        elif file.endswith('.tif') and 'rhos_833' in file:
            print(f'Found nir band: {file}')
            nirInput =  os.path.join(directory, file)


# %% - RGB composite
if create_rbg_composite:
    if not os.path.exists(directory + '\_RGB'):
        os.makedirs(directory + '\_RGB')
    
    rgb_compTF, rgb_composite = sdb.composite(rgb, os.path.join(directory + '\_RGB', os.path.basename(directory) + '_rgb_composite.tif'))
    
    if rgb_compTF:
        print(f'\nWrote: {rgb_composite}')
    else:
        print('\nCreating RGB composite failed...')


# %% - Workflow

if full_workflow:
    

    # %% - band masking
    # returns of list of masked bands for each wavelength
    mTF, maskOutput, out_meta = sdb.mask_imagery(redInput, greenInput, blueInput, nirInput, maskSHP, predictor_dir)
    
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
    
    # Green SDB (deeper)
    gTF, outraster_name, pSDBg, out_meta = sdb.pSDBn(maskedBlue, maskedGreen, rol_name='pSDBg', output_dir=predictor_dir)
    if gTF:
        pSDBg_name = outraster_name
    else:
        print("No green SDB raster dataset was returned from the pSDBgreen function.")
    
    # Red SDB (shallower)
    rTF, outraster_name, pSDBr, out_meta = sdb.pSDBn(maskedBlue, maskedRed, rol_name='pSDBr', output_dir=predictor_dir)
    if rTF:
        pSDBr_name = outraster_name
    else:
        print("No green SDB raster dataset was returned from the pSDBgreen function.")
    
    
    # %% - surface roughness predictors
    
    # GDAL sLope -- pSDBg_name is a string
    slopeTF, pSDBg_slope = sdb.slope(pSDBg_name, slope_name='pSDBg_slope.tif', output_dir=predictor_dir)
    if slopeTF:
        print(f'\nWrote: {pSDBg_slope}')
    else:
        print('\nCreating pSDBg slope failed...')
    
    # https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    window_stdevslopeTF, pSDBg_stdevslope = sdb.window_stdev(pSDBg_slope, window=7, stdev_name='pSDBg_stdevslope.tif', output_dir=predictor_dir, out_meta=out_meta)
    if window_stdevslopeTF:
        print(f'\nWrote: {pSDBg_stdevslope}')
    else:
        print('\nCreating pSDBg stdev slope failed...')
    
    # stdevslopeTF, pSDBg_stdevslope = sdb.stdev(pSDBg_slope, window=7, stdev_name='pSDBg_stdevslope.tif', output_dir=predictor_dir, out_meta=out_meta)
    # if stdevslopeTF:
    #     print(f'\nWrote: {pSDBg_stdevslope}')
    # else:
    #     print('\nCreating pSDBg stdev slope failed...')
    
    # curvature RichDEM
    # sys.stdout = open(os.devnull, 'w')
    # curTF, pSDBg_curve = sdb.curvature(pSDBg, curvature_name='pSDBg_curvature.tif', output_dir=predictor_dir, out_meta=out_meta)
    # # sys.stdout = sys.__stdout__
    # if curTF:
    #     print(f'\nWrote: {pSDBg_curve}')
    # else:
    #     print('\nCreating pSDBg curvature failed...')
    
    # # GDAL Terrain Ruggedness Index (TRI)
    # https://gdal.org/programs/gdaldem.html
    # https://gdal.org/api/python/osgeo.gdal.html
    # triTF, pSDBg_tri = sdb.tri(pSDBg_name, tri_name='pSDBg_tri_Wilson.tif', output_dir=predictor_dir)
    # if triTF:
    #     print(f'\nWrote: {pSDBg_tri}')
    
    # # GDAL TPI
    # tpiTF, pSDBg_tpi = sdb.tpi(pSDBg_name, tpi_name='pSDBg_tpi.tif', output_dir=predictor_dir)
    # if tpiTF:
    #     print(f'\nWrote: {pSDBg_tpi}')
        
    # GDAL Roughness
    roughTF, pSDBg_roughness = sdb.roughness(pSDBg_name, roughness_name='pSDBg_roughness.tif', output_dir=predictor_dir)
    if roughTF:
        print(f'\nWrote: {pSDBg_roughness}')
        
    # # Skimage canny edge detection
    # # cTF, blue_canny = sdb.canny(maskedBlue, canny_name='blue_canny.tif', output_dir=predictor_dir, out_meta=out_meta)
    # canTF, pSDBg_canny = sdb.canny(pSDBg, canny_name='pSDBg_canny.tif', output_dir=predictor_dir, out_meta=out_meta)
    # if canTF:
    #     print(f'\nWrote: {pSDBg_canny}')
        
    # sobelTF, pSDBg_sobel, sobel_edges = sdb.sobel_filt(pSDBg, sobel_name='pSDBg_sobel.tif', output_dir=predictor_dir, out_meta=out_meta)
    # if sobelTF:
    #     print(f'\nWrote: {pSDBg_sobel}')
        
    # prewittTF, pSDBg_prewitt, prewitt_edges = sdb.prewitt_filt(pSDBg, prewitt_name='pSDBg_prewitt.tif', output_dir=predictor_dir, out_meta=out_meta)
    # if prewittTF:
    #     print(f'\nWrote: {pSDBg_prewitt}')
        
    # test
    # stdevslopeTF, pSDBg_stdevslope, stdev_slope = sdb.stdev_slope(pSDBg, window_size=5, stdev_name='pSDBg_stdevslope_2.tif', output_dir=predictor_dir, out_meta=out_meta)
        
    # should probably change to write file in a function
    # rugosity worth continueing? 
    
    
    # %% - predictor composite
    print('\nBuilding compsite image...\n')
    
    # dir_list = [os.path.join(predictor_dir, file) for file in os.listdir(predictor_dir) if file.endswith(".tif") and 'masked' not in file]
    dir_list = [os.path.join(predictor_dir, file) for file in os.listdir(predictor_dir) if file.endswith(".tif") and 'pSDBg_slope' not in file]
    
    
    composite_dir = predictor_dir + '\_Composite'
    
    if not os.path.exists(composite_dir):
        os.makedirs(composite_dir) 
    
    output_composite = os.path.join(composite_dir, composite_name)
    
    compTF, composite = sdb.composite(dir_list, output_composite)
    
    if compTF:
        print(f'\nWrote: {composite}')
    else:
        print('\nCreating composite failed...')
    
        
t.toc()
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
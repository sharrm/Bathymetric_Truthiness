"""
@author: sharrm

Updated: 21Mar2023
"""

# Identify user defined function files
import SDB_Functions as sdb
import os, sys
import datetime, time
# from pytictoc import TicToc
# import linear_regression as slr


# %% - globals
current_time = datetime.datetime.now() # current time for output file names
start_time = time.time() # start time for process timing


# %% - features class to build feature composite
class Features:   
    """
    workflow
    1. mask bands (can merge steps 1 and 2 in a function)
    2. compute features
    3. build composite of features
    """
    
    ## - init class    
    def __init__(self, rgbnir_dir, maskSHP, num_options):
        self.rgbnir_dir = rgbnir_dir
        self.maskSHP = maskSHP
        self.num_options = num_options
        self.composite_features_list = []
        
        self.feature_dir = self.rgbnir_dir + '\_Features_' + self.num_options + 'Bands' 
       
        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir)
    
    ## - band masking
    def mask_to_aoi(self, band, outraster_name): # this should handle the less general functionality of dealing with rgb nir
        mTF, mask_raster_name = sdb.mask_imagery(band, self.maskSHP, outraster_name) # this should be a more general function
        
        if mTF:
            print(f'--Masked file output to {mask_raster_name}')
        else:
            print('Issue masking RGB NIR features.')
        return mask_raster_name
        
    ## - ratio between bands / opitcally deep index 1
    def ODI_1(self, blue_name, green_name, odi1_name):
        # odi 1
        odi1TF, odi_name = sdb.odi_1(blue_name, green_name, odi_1_name=odi1_name, output_dir=self.feature_dir)
               
        if odi1TF:
            print(f'--ODI 1 output to {odi_name}')
        else:
            print("Issue generating optically deep index (1)...")
        return odi_name
    
    ## - ratio between bands / opitcally deep index 1
    def ODI_2(self, blue_name, green_name, odi2_name):
        # odi 2
        odi2TF, odi_name = sdb.odi_2(blue_name, green_name, odi_2_name=odi2_name, output_dir=self.feature_dir)
               
        if odi2TF:
            print(f'--Normalized difference output to {odi_name}')
        else:
            print("Issue generating normalized difference...")
        return odi_name
    
    ## - ratio of logs between bands / relative bathymetry
    def pSDB(self, band1, band2, rol_name):
        # pSDB
        pSDBTF, pSDB_name = sdb.pSDBn(band1, band2, rol_name=rol_name, output_dir=self.feature_dir)
               
        if pSDBTF:
            print(f'--Band ratio output to {pSDB_name}')
        else:
            print("Issue generating band ratios...")
        return pSDB_name
    
    ## - surface roughness predictors
    def _slope(self, pSDBg_name):
        # GDAL sLope -- pSDBg_name is a string
        pSDB_slope_name = pSDBg_name.replace('.tif', '_slope.tif')
        slopeTF, pSDB_slope = sdb.slope(pSDBg_name, slope_name=pSDB_slope_name, output_dir=self.feature_dir)
        if slopeTF:
            print(f'----Output temp slope: {pSDB_slope_name}')
        else:
            print(f'\nCreating {pSDB_slope_name} failed...')
        return pSDB_slope_name
    
    ## - roughness
    def pSDBg_surface_roughness(self, pSDBg_name, window_size):
        pSDBg_slope_name = Features._slope(self, pSDBg_name)
        
        # https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
        window_stdevslopeTF, pSDBg_stdevslope_name = sdb.window_stdev(pSDBg_slope_name, window=window_size, 
                                                                 stdev_name='pSDBg_stdevslope.tif', 
                                                                 output_dir=self.feature_dir)
        if window_stdevslopeTF:
            print(f'--pSDBg standard deviation of slope output to {pSDBg_stdevslope_name}')
        else:
            print('\nCreating pSDBg stdev slope failed...')
        
        # GDAL Roughness
        roughTF, pSDBg_roughness_name = sdb.roughness(pSDBg_name, roughness_name='pSDBg_roughness.tif', output_dir=self.feature_dir)
        if roughTF:
            print(f'--pSDBg roughness output to {pSDBg_roughness_name}')
        else:
            print('\nCreating pSDBg roughness failed...')
        return pSDBg_stdevslope_name, pSDBg_roughness_name
    
    ## - predictor composite
    def build_feature_composite(self, num_bands):
        print('\nBuilding compsite image...')
        
        composite_dir = self.feature_dir + '\_Composite'
        
        if not os.path.exists(composite_dir):
            os.makedirs(composite_dir)
        
        out_composite_name = os.path.join(composite_dir, 
                                          os.path.basename(self.maskSHP).replace('.shp','')
                                          + f'_{num_bands}Bands_composite_' 
                                          + current_time.strftime('%Y%m%d_%H%M') 
                                          + '.tif')
        
        compTF, band_number = sdb.composite(self.composite_features_list, out_composite_name)
        
        if compTF:
            print(f'--Output Composite: {out_composite_name} with {band_number} bands')
        else:
            print('\nCreating composite failed...')
        return out_composite_name
    
    def feature_options(self, blue, green, red, nir, cmyk, hsv, odi_1, odi_2, ndwi, pSDBg, pSDBr, pSDBg_roughness, window_size): # options to pass in
        # list of bands
        rasters = os.listdir(self.rgbnir_dir)

        # dict to store surface reflectance bands by wavelength
        surface_reflectance = {}
        
        for band in rasters:
            try:
                # simply customizing the output filenames here -- there's probably a better method
                if '492' in band or 'B2' in band: # blue wavelength (492nm)
                    surface_reflectance['blue name'] = os.path.join(self.feature_dir, 
                                                               'masked_' + os.path.basename(band)[-7:-4] + '.tif')
                    surface_reflectance['blue band'] = os.path.join(self.rgbnir_dir, band)
                elif '560' in band or 'B3' in band or '559' in band: # green wavelength (560nm)
                    surface_reflectance['green name'] = os.path.join(self.feature_dir, 
                                                           'masked_' + os.path.basename(band)[-7:-4] + '.tif')
                    surface_reflectance['green band'] = os.path.join(self.rgbnir_dir, band)
                elif '665' in band or 'B4' in band: # red wavelength (665nm)
                    surface_reflectance['red name'] = os.path.join(self.feature_dir, 
                                                         'masked_' + os.path.basename(band)[-7:-4] + '.tif')
                    surface_reflectance['red band'] = os.path.join(self.rgbnir_dir, band)
                elif '833' in band or 'B8' in band: # red wavelength (665nm)
                    surface_reflectance['nir name'] = os.path.join(self.feature_dir, 
                                                         'masked_' + os.path.basename(band)[-7:-4] + '.tif')
                    surface_reflectance['nir band'] = os.path.join(self.rgbnir_dir, band)
            except:
                print('Expected image name ending with wavelength in nanometers (i.e., 492, 560, 665, 833) or band number (e.g., B1, B2, ...) in file name.')

        features_list = []
        
        # general workflow setup
        print('\nGenerating features...')
        # need to modify conditions in case we don't want blue, green, red etc in composite but are needed for ratios etc
        blue_aoi = Features.mask_to_aoi(self, surface_reflectance['blue band'], surface_reflectance['blue name'])
        green_aoi = Features.mask_to_aoi(self, surface_reflectance['green band'], surface_reflectance['green name'])
        red_aoi = Features.mask_to_aoi(self, surface_reflectance['red band'], surface_reflectance['red name'])
        nir_aoi = Features.mask_to_aoi(self, surface_reflectance['nir band'], surface_reflectance['nir name'])
        
        if blue:
            self.composite_features_list.append(blue_aoi)
            features_list.append("'Blue Band',")
        if green:
            self.composite_features_list.append(green_aoi)
            features_list.append("'Green Band',")
        if red:
            self.composite_features_list.append(red_aoi)
            features_list.append("'Red Band',")
        if nir:
            self.composite_features_list.append(nir_aoi)
            features_list.append("'NIR Band',")
        if cmyk:
            cmykTF, cyan_aoi, magenta_aoi, yellow_aoi, black_aoi = sdb.rgb_to_cmyk(red_aoi, green_aoi, blue_aoi, output_dir=self.feature_dir)
            self.composite_features_list.append(cyan_aoi)
            # self.composite_features_list.append(magenta_aoi)
            # self.composite_features_list.append(yellow_aoi)
            self.composite_features_list.append(black_aoi)
            features_list.append("'Cyan',")
            # features_list.append("'Magenta',")
            # features_list.append("'Yellow',")
            features_list.append("'Black',")
        if hsv:
            hsvTF, hue_aoi, saturation_aoi, value_aoi = sdb.rgb_to_hsv(red_aoi, green_aoi, blue_aoi, output_dir=self.feature_dir)
            self.composite_features_list.append(hue_aoi)
            self.composite_features_list.append(saturation_aoi)
            self.composite_features_list.append(value_aoi)
            features_list.append("'Hue',")
            features_list.append("'Saturation',")
            features_list.append("'Value',")
        if odi_1:
            odi_1 = Features.ODI_1(self, blue_aoi, green_aoi, odi1_name='odi_1.tif')
            self.composite_features_list.append(odi_1)
            features_list.append("'ODI 1',")
        if odi_2:
            odi_2 = Features.ODI_2(self, blue_aoi, green_aoi, odi2_name='odi_1.tif')
            self.composite_features_list.append(odi_2)
            features_list.append("'ODI 2',")
        if ndwi:
            ndwi_aoi = Features.ODI_2(self, nir_aoi, green_aoi, odi2_name='ndwi.tif')
            self.composite_features_list.append(ndwi_aoi)
            features_list.append("'NDWI',")
        if pSDBg:
            pSDBg_aoi = Features.pSDB(self, blue_aoi, green_aoi, rol_name='pSDBg.tif')
            self.composite_features_list.append(pSDBg_aoi)
            features_list.append("'pSDBg',")
        if pSDBr:
            pSDBr_aoi = Features.pSDB(self, blue_aoi, red_aoi, rol_name='pSDBr.tif')
            self.composite_features_list.append(pSDBr_aoi)
            features_list.append("'pSDBr',")
        if pSDBg_roughness and pSDBg:
            pSDBg_stdevslope_aoi, pSDBg_roughness_aoi = Features.pSDBg_surface_roughness(self, pSDBg_aoi, window_size)
            self.composite_features_list.append(pSDBg_stdevslope_aoi)
            self.composite_features_list.append(pSDBg_roughness_aoi)
            features_list.append("'pSDBg Standard Deviation of Slope',")
            features_list.append("'pSDBg Roughness'")
        
        num_bands = len(features_list)
        
        if num_bands != self.num_options:
            print('Found unexpected number of features in output composite.')
        
        out_composite_name = Features.build_feature_composite(self, num_bands)
        
        return out_composite_name, features_list

    ## - end of Features class
    
    
# %% - rgb composite
def rgb_composite(rgb_dir): 
    print('Creating RGB composite...')
    rgb = []

    if not os.path.exists(rgb_dir + '\_RGB'):
        os.makedirs(rgb_dir + '\_RGB')

    for input_band in os.listdir(rgb_dir):
        try:
            if input_band.endswith('.tif') and 'rhos_492' in input_band:
                print(f'Found blue band: {input_band}')
                rgb.append(os.path.join(rgb_dir, input_band))
            elif input_band.endswith('.tif') and 'rhos_560' in input_band or 'rhos_559' in input_band:
                print(f'Found green band: {input_band}')
                rgb.append(os.path.join(rgb_dir, input_band))
            elif input_band.endswith('.tif') and 'rhos_665' in input_band:
                print(f'Found red band: {input_band}')
                rgb.append(os.path.join(rgb_dir, input_band))
            # elif input_band.endswith('.tif') and 'rhos_833' in input_band:
            #     print(f'Found nir band: {input_band}')
            #     nirInput =  os.path.join(rgb_dir, input_band)
        except:
            print('May be issue with input file name. Expected ".tif" and "rhos_(wavelength)" in name.')
            
    rgb_compTF, rgb_composite_name, band_number = sdb.composite(rgb, os.path.join(rgb_dir + '\_RGB', 
                                                                                  os.path.basename(rgb_dir) + '_rgb_composite.tif'))
    
    if rgb_compTF:
        print(f'--Wrote: {rgb_composite_name} with {band_number} bands.')
    else:
        print('\nCreating RGB composite failed...')
    return None


# %% - main
def main():    
    # input imagery
    # train_test = [r"P:\Thesis\Training\_New_Feature_Building"] # training
    train_test = [r"P:\Thesis\Test Data\_Turbid_Tests", r"P:\Thesis\Test Data\_New_Feature_Building"] # testing
    
    img_dirs = []
    for loc in train_test:
        [img_dirs.append(os.path.join(loc, folder)) for folder in os.listdir(loc)]
    
    # input aoi
    # extent_dir = [r"P:\Thesis\Extents\_Training"] # training    
    extent_dir = [r"P:\Thesis\Extents\_Turbid_Tests", r"P:\Thesis\Extents\_Testing"] # testing
    
    maskSHP_dir = []
    for loc in extent_dir:
        [maskSHP_dir.append(os.path.join(loc, shp)) for shp in os.listdir(loc) if shp.endswith('.shp')]

    ft_options = {'blue': True, 
                  'green':True, 
                  'red':  True, 
                  'nir':  True,
                  'cmyk': True,
                  'hsv': True,
                  'odi_1': True,
                  'odi_2': True,
                  'ndwi': True,
                  'pSDBg': True,
                  'pSDBr': True,
                  'pSDBg_roughness':True,
                  'window_size':7}
    
    num_options = str(sum(ft_options.values()) - (ft_options['window_size']) + 1 )# exclude window size and include two for roughness
    if not ft_options['pSDBg_roughness']:
        num_options = str(sum(ft_options.values()) - (ft_options['window_size']) + 1 )
    elif ft_options['cmyk']:
        num_options = str(sum(ft_options.values()) - (ft_options['window_size']) + 4 ) # 2 without hsv
    
    final_composite_list = []

    for rgbnir_dir in img_dirs:
        for maskSHP in maskSHP_dir:
            if sdb.check_bounds(rgbnir_dir, maskSHP):
                print('Creating features based on:')
                print(f'--Imagery: {rgbnir_dir}')
                print(f'--AOI: {maskSHP}')
                                
                fts = Features(rgbnir_dir, maskSHP, num_options)
                composite_name, features = fts.feature_options(**ft_options)
                final_composite_list.append(composite_name)

                print('--Features in composite:')
                [print(feature) for feature in features]
                print('------------------------------------------------------')
            else:
                continue
    
    print(f'\nFinal composite list: {final_composite_list}')

if __name__ == '__main__':
    start = current_time.strftime('%H:%M:%S')
    print(f'Starting at {start}\n')
    model = main()
    runtime = time.time() - start_time
    print(f'\nTotal elapsed time: {runtime:.1f} seconds / {(runtime/60):.1f} minutes')


# %% - regression with reference data set

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
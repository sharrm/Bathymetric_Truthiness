# Bathymetric Truthiness
#

# List of scripts:
# - feature_building
# - SDB_Functions
# - rf_prediction
# - rf_training
#

# Description:
# feature_building: Takes surface reflectance inputs and generates an n-dimensional geotiff of input features for use in the rf_training or rf_prediction scripts. The main purpose of this script is to control functions in SDB_Functions, and more easily organize the code.
# SDB_Functions: Main function for performing masking, pSDBg, pSDBr, surface roughness, and optically shallow index calculations
# rf_training: Uses input geotiff from the feature_building script to train a random forest model. Optionally can compute validation/performance metrics and plot results.
# rf_prediction: Uses input geotiff from feature_building script to make a prediction, using the trained model. Optionally can compute test performance metrics and save/plot results.
#

# Utilities and U_Net folders contain other scripts used in earlier stages of workflow development

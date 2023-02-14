# -*- coding: utf-8 -*-
"""
Last updated Nov/03/2022

@ author: Sreenivas Bhattiprolu, ZEISS
@ modified by: Jaehoon Jung, PhD, OSU

Semantic segmentation using U-Net architecture (prediction)

@ modified by Matt Sharr
"""

from patchify import patchify, unpatchify

import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
import keras


def plotPatches(im,row):
    plt.figure(figsize=(9, 9))
    square = im.shape[1]
    ix = 1
    for i in range(square):
    	for j in range(square):
    		ax = plt.subplot(square, square, ix)
    		ax.set_xticks([])
    		ax.set_yticks([])
    		plt.imshow(im[i+row, j, :, :], cmap='jet')
    		ix += 1
    
def padding(image,s_patch):
    h,w = np.shape(image)
    pad_row = (0, s_patch - (h % s_patch))
    pad_col = (0, s_patch - (w % s_patch))
    image = np.pad(image, [pad_row, pad_col], mode='constant', constant_values=0)
    return image,h,w

# def padding3D(image,s_patch):
#     h,w,d = np.shape(image)
#     pad_row = (0, s_patch - (h % s_patch))
#     pad_col = (0, s_patch - (w % s_patch))
#     image = np.pad(image, [pad_row, pad_col, (0,0)], mode='constant', constant_values=0)
#     return image,h,w

def padding3D(image,s_patch):
    # https://stackoverflow.com/questions/50008587/zero-padding-a-3d-numpy-array
    h,w,d = np.shape(image)
    pad_row = (0, s_patch - (h % s_patch))
    pad_col = (0, s_patch - (w % s_patch))
    dim = [pad_row, pad_col]
    for i in range(2,len(image.shape)):
        dim.append((0,0))    
    image = np.pad(image, dim, mode='constant', constant_values=0)
    return image,h,w

def plotImage(zi,t_cmap, filename):
    plt.figure()
    plt.clf()
    plt.imshow(zi, cmap=t_cmap)
    plt.colorbar()
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    # plt.axis('off')
    plt.title(filename, fontsize=11)
    # plt.savefig('Z:\\CE560\\HW3\\Report\\Images\\' + filename + '.png', dpi=300)
    plt.show()

def scaleData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# %% hyper parameters
s_patch = 128
s_step = 128
n_classes = 3

# %% load data and model
# o_image = tiff.imread(r"P:\Thesis\Test Data\Puerto Real\_10Band\_Composite\Puerto_Real_Smaller_composite.tif")
o_image = tiff.imread(r"P:\Thesis\Test Data\TinianSaipan\_10Band\_Composite\Saipan_Extents_composite.tif")

o_image = np.nan_to_num(o_image)
o_image[o_image == -9999] = 0
# image = image.astype(np.float) / 255.  # scale dataset
# image = scaleData(o_image.astype(np.float)) # scale data

image = np.zeros(np.shape(o_image)) # default float64 (scaled array)
for i in range(0, np.shape(image)[2]):
    ft = o_image[:,:,i]
    # ft[(ft < np.mean(ft) - 3 * np.std(ft)) | (ft > np.mean(ft) + 3 * np.std(ft))] = 0
    image[:,:,i] = scaleData(ft.astype(np.float64))

image, h, w = padding3D(image,s_patch)
# image, h, w = padding(image,s_patch)

# Trained U-Net model
model = keras.models.load_model(r"P:\Thesis\Models\UNet\Unet_model_10bands_128step20230213_1431.hdf5", compile=False)

# %% prediction, patch by patch 
# patches = patchify(image, (s_patch, s_patch), step = s_step)#-- split image into small patches with overlap  
#-- plotPatches(patches,20) # plot patches
patches = patchify(image, (s_patch, s_patch, image.shape[2]), step=s_step) 
row, col, dep, hi, wi, d = patches.shape
patches = patches.reshape(row*col*dep, hi, wi, d)  

# o image size: (1734, 1730, 10)
# image size: (1792, 1792, 10)

patches_predicted = [] # store predicted images (196,128,128)
for i in range(patches.shape[0]): # loop through all patches
    print("Now predicting on patch: ", i)
    patch1 = patches[i,:,:,:] # all rows, cols, dep of each patch (128,128,10)
    patch1 = np.expand_dims(np.array(patch1), axis=[0]) # expand first dimension to fit into model (1,128,128,10)
    patch1_prediction = model.predict(patch1) # predict on patch (1,128,128,3)
    patch1_predicted_img = np.argmax(patch1_prediction, axis=3)[0,:,:] # along class axis store maximum values (128,128)
    patches_predicted.append(patch1_predicted_img) # store patch prediction in list (196 in total)

patches_predicted = np.array(patches_predicted) # create array of each patch in list (196,128,128)
# need to figure out reshaping to 3D, this code works for reshaping from 2D
# reshaped size (196, 128, 128, 1)
patches_predicted_reshaped = np.reshape(patches_predicted, (row, col, s_patch, s_patch) ) #-- Gives a new shape to an array without changing its data
image_predicted = unpatchify(patches_predicted_reshaped, image.shape[0:2]) #-- merge patches into original image
image_predicted = image_predicted[:h,:w] #-- recover original image size
# #-- plot segmented image
# plotImage(o_image[0:2],'Greys_r', 'Original')
plotImage(image_predicted,'viridis', 'Classified')

# classified image vs point cloud intensity

# patches_predicted_reshaped = np.reshape(patches_predicted, (image.shape[0], -1))
# plt.imshow(patches_predicted_reshaped)
# plt.colorbar()
# plt.show()

# image_height, image_width, channel_count = image.shape # (1734,1730,10)
# output_height = image_height - (image_height - s_patch) % s_step # [1734 - (1734 - 128) % 128] = 1664
# output_width = image_width - (image_width - s_patch) % s_step # [1730 - (1730 - 128) % 128] = 1664
# output_shape = (output_height, output_width) # (1664, 1664, 10)

#%%

# https://stackoverflow.com/questions/68249421/how-to-modify-patches-made-by-patchify

# for row in range(patches.shape[0]):
#     # for col in range(patches.shape[1]):
#     print("Now predicting on patch", row, col)
#     patch1 = patches[row,:,:,:]
#     patch1 = np.expand_dims(np.array(patch1), axis=[0])
#     patch1_prediction = model.predict(patch1)
#     patch1_predicted_img = np.argmax(patch1_prediction, axis=3)[0,:,:]
#     patches_predicted.append(patch1_predicted_img)

# for row in range(patches.shape[0]):
#     for col in range(patches.shape[1]):
#         print("Now predicting on patch", row, col)
#         patch1 = patches[row,col,:,:]
#         patch1 = np.expand_dims(np.array(patch1), axis=[0,3])
#         patch1_prediction = model.predict(patch1)
#         patch1_predicted_img = np.argmax(patch1_prediction, axis=3)[0,:,:]
#         patches_predicted.append(patch1_predicted_img)



# https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
# image_height, image_width, channel_count = image.shape
# patch_height, patch_width, step = 128, 128, 128
# patch_shape = (patch_height, patch_width, channel_count)
# patches = patchify(image, patch_shape, step=step)

# output_patches = np.empty(patches.shape).astype(np.uint8)
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         patch = patches[i, j, 0]
#         output_patch = model.predict(patch)  # process the patch
#         output_patches[i, j, 0] = output_patch

# image_height, image_width, channel_count = o_image.shape # (1734,1730,10)
# output_height = image_height - (image_height - s_patch) % s_step # [1734 - (1734 - 128) % 128] = 1664
# output_width = image_width - (image_width - s_patch) % s_step # [1730 - (1730 - 128) % 128] = 1664
# output_shape = (output_height, output_width, channel_count) # (1664, 1664, 10)
# output_image = unpatchify(output_patches, output_shape) 











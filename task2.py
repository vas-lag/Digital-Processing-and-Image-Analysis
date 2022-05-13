# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 01:55:56 2021

@author: Billy
"""

import numpy as np
import scipy.io
import scipy.fftpack
import matplotlib.pyplot as plt

def read_image(path):
    mat = scipy.io.loadmat(path)
    return mat

def split_to_parts(im):
    parts = []
    lenX, lenY = im.shape[0], im.shape[1]
    for x in range(0, lenX, 32):
        for y in range(0, lenY, 32):
            parts.append(np.copy(im[x:x+32, y:y+32]))
    return parts

def merge_from_parts(parts):
    length = int(np.sqrt(len(parts)))
    im = np.zeros((length * 32, length * 32))
    for x in range(length):
        for y in range(length):
            im[x * 32:x * 32 + 32, y * 32:y * 32 + 32] = parts[x * length + y]
    return im
    

def dct(images):
    res = []
    for im in images:
        res.append(dct_row_col(im))
    return res

def idct(imagesdct):
    res = []
    for im in imagesdct:
        res.append(idct_row_col(im))
    return res

def dct_row_col(im):
    imdct = np.zeros(im.shape, dtype='float64')
    #rows first
    for row in range(im.shape[0]):
        imdct[row, :] = scipy.fftpack.dct(im[row, :])
    
    #then columns
    for col in range(im.shape[1]):
        imdct[:, col] = scipy.fftpack.dct(imdct[:, col])
    return imdct

def idct_row_col(imdct):
    im = np.zeros(imdct.shape, dtype='float64')
    #rows first
    for row in range(im.shape[0]):
        im[row, :] = scipy.fftpack.idct(imdct[row, :])
    
    #then columns
    for col in range(im.shape[1]):
        im[:, col] = scipy.fftpack.idct(im[:, col])
    return im

def quantize_dct_threshold(r, parts):
    parts_copy = parts.copy()
    for idx, im in enumerate(parts_copy):
        threshold = np.quantile(np.abs(im), 1 - r)
        #discard any dct component with absolute value smaller than the threshold
        im_copy = np.copy(im)
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if abs(im_copy[x, y]) < threshold:
                    im_copy[x, y] = 0
        parts_copy[idx] = im_copy
    return parts_copy

def quantize_dct_zone(r, parts):
    parts_copy = parts.copy()
    arr = np.zeros((parts[0].shape[0], parts[0].shape[1], len(parts)))
    for partIndex in range(len(parts)):
        arr[:, :, partIndex] = parts[partIndex]
    var = np.var(arr, (2))
    threshold = np.quantile(var, 1 - r)
    for idx, im in enumerate(parts_copy):
        im_copy = np.copy(im)
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if var[x, y] < threshold:
                    im_copy[x, y] = 0
        parts_copy[idx] = im_copy
    return parts_copy

def plot_g(im, title=None):
    plt.figure()
    plt.imshow(im, cmap='gray')
    if title:
        plt.title(title)
    

r = 0.05
mat = read_image('../images/2/flower.mat')
im = mat["flower"]
plot_g(im, 'original image')
image_parts = split_to_parts(im)
image_parts_dct = dct(image_parts)

image_parts_dct_quantized_threshold = quantize_dct_threshold(r, image_parts_dct)
image_parts_reconstructed = idct(image_parts_dct_quantized_threshold)
image = merge_from_parts(image_parts_reconstructed)
image = image / (np.max(image) / np.max(im))
plot_g(image, f'compressed using {r * 100}% of dct coefficients')

image_parts_dct_quantized_zone = quantize_dct_zone(r, image_parts_dct)
image_parts_reconstructed = idct(image_parts_dct_quantized_zone)
image = merge_from_parts(image_parts_reconstructed)
image = image / (np.max(image) / np.max(im))
plot_g(image, f'compressed using {r * 100}% of dct coefficients')

# mean_error_threshold = []
# mean_error_zone = []
# for r in np.arange(0.05, 0.5, 0.02):
#     image_parts_dct_quantized_threshold = quantize_dct_threshold(r, image_parts_dct)
#     image_parts_reconstructed = idct(image_parts_dct_quantized_threshold)
#     image = merge_from_parts(image_parts_reconstructed)
#     image = image / (np.max(image) / np.max(im))
#     mean_error_threshold.append(np.mean((im - image) ** 2))
    
#     image_parts_dct_quantized_zone = quantize_dct_zone(r, image_parts_dct)
#     image_parts_reconstructed = idct(image_parts_dct_quantized_zone)
#     image = merge_from_parts(image_parts_reconstructed)
#     image = image / (np.max(image) / np.max(im))
#     mean_error_zone.append(np.mean((im - image) ** 2))
# plt.figure()
# plt.plot(np.arange(0.05, 0.5, 0.02), mean_error_threshold, label='theshold method')
# plt.plot(np.arange(0.05, 0.5, 0.02), mean_error_zone, label='zone method')
# plt.title('Mean square error')
# plt.legend()


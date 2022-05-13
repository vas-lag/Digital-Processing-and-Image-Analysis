# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:32:32 2021

@author: Billy
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_image(path):
    im = Image.open(path)
    im = np.array(im)
    return im

def use_all_dynamic_area(im):
    maxElement = np.max(im)
    minElement = np.min(im)
    
    # the following equation shifts our image to [0, 255]
    im = (im - minElement) / (maxElement - minElement) * 255 + 0
    im = im.astype(np.uint8)
    return im

def fft_row_col(im):
    imfft = np.zeros(im.shape, dtype='complex')
    #rows first
    for row in range(im.shape[0]):
        imfft[row, :] = np.fft.fft(im[row, :])
    
    #then columns
    for col in range(im.shape[1]):
        imfft[:, col] = np.fft.fft(imfft[:, col])
    return imfft

def inverse_fft_row_col(imfft):
    im = np.zeros(imfft.shape, dtype='complex')
    #rows first
    for row in range(imfft.shape[0]):
        im[row, :] = np.fft.ifft(imfft[row, :])
    
    #then columns
    for col in range(imfft.shape[1]):
        im[:, col] = np.fft.ifft(im[:, col])
    return im

def hamming2d(r, size):
    ham = np.hamming(size)[:,None] # 1D hamming
    ham2d = np.sqrt(np.dot(ham, ham.T)) ** r # expand to 2D hamming
    return ham2d

def create_low_pass(t, shape):
    filt = np.zeros(shape)
    midx, midy = shape[0] / 2, shape[1] / 2
    for x in range(shape[0]):
        for y in range(shape[1]):
            if np.sqrt((x - midx) ** 2 + (y - midy) ** 2) < t:
                filt[x, y] = 1
    return filt

def plot_g(im, title=None):
    plt.figure()
    plt.imshow(im, cmap='gray')
    if title:
        plt.title(title)


#read image and expand to use all dynamic area
im = read_image('../images/1/aerial.tiff')
im = use_all_dynamic_area(im)

#shift zero component to the center of the image
filt = np.zeros(im.shape)
for row in range(im.shape[0]):
    for col in range(im.shape[1]):
        filt[row, col] = (-1) ** (row + col + 2)
im = im * filt

#calculate and plot fft of the image
imfft = fft_row_col(im)
plot_g(np.abs(imfft), 'Linear magnitude of image DFT')
plot_g(np.log(np.abs(imfft)), 'logarithmic magnitude of image DFT')

#our 2d lowpass filter
# low_pass = create_low_pass(60, imfft.shape)
ham2d = hamming2d(10, imfft.shape[0])
plot_g(ham2d, 'low-pass filter in frequency domain')

#pass our image through the filter in frequency domain
imfftlow = imfft * ham2d
plot_g(np.log(np.abs(imfftlow)), 'frequency domain of the image after the filter')

#calculate the ifft of the image and show the result
imlow = inverse_fft_row_col(imfftlow)
imlow = imlow * filt
imlow = np.rint(np.abs(imlow)).astype(np.uint8)
plot_g(imlow)
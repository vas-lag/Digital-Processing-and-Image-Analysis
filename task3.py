# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:52:36 2021

@author: Billy
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
from PIL import Image

def read_image(path):
    im = Image.open(path)
    im = np.array(im)
    return im

def plot_g(im, title=None):
    plt.figure()
    plt.imshow(im, cmap='gray')
    if title:
        plt.title(title)
    
def add_gaussian_noise_snr(im, snr):
    mean = np.mean(im) / 255
    linsnr = 10 ** (snr / 10)
    var = mean / linsnr
    im = skimage.util.random_noise(im, var=var) * 255
    return im.astype(np.uint8)
    
def add_s_and_p_noise(im, percetnage):
    im = skimage.util.random_noise(im, mode='s&p', amount=percetnage) * 255
    return im.astype(np.uint8)

def moving_average_filter(im, block):
    pad = block // 2
    copy = np.zeros((im.shape[0] + 2 * pad, im.shape[1] + 2 * pad))
    copy[pad:-pad, pad:-pad] = np.copy(im)
    filtered = np.copy(im)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            filtered[x, y] = np.mean(copy[x:x+2*pad+1, y:y+2*pad+1])
    return filtered

def median_filter(im, block):
    pad = block // 2
    copy = np.zeros((im.shape[0] + 2 * pad, im.shape[1] + 2 * pad))
    copy[pad:-pad, pad:-pad] = np.copy(im)
    filtered = np.copy(im)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            filtered[x, y] = np.median(copy[x:x+2*pad+1, y:y+2*pad+1])
    return filtered


im= read_image('../images/3/clock.tiff')
plot_g(im, 'original image')
im_gaussian = add_gaussian_noise_snr(im, 15)
plot_g(im_gaussian, 'image with white gaussian noise')
im_restored_mean = moving_average_filter(im_gaussian, 3)
plot_g(im_restored_mean, 'image restored using moving average filter')
im_restored_median = median_filter(im_gaussian, 3)
plot_g(im_restored_median, 'image restored using median filter')

im_s_and_p = add_s_and_p_noise(im, 0.05)
plot_g(im_s_and_p, 'image with salt & pepper noise')
im_restored_mean = moving_average_filter(im_s_and_p, 3)
plot_g(im_restored_mean, 'image restored using moving average filter')
im_restored_median = median_filter(im_s_and_p, 3)
plot_g(im_restored_median, 'image restored using median filter')

im_with_noise = add_s_and_p_noise(im_gaussian, 0.2)
plot_g(im_with_noise, 'image with both noise types')
im_restored_partial = median_filter(im_with_noise, 3)
im_restored = moving_average_filter(im_restored_partial, 3)
plot_g(im_restored, 'restored image using both filters')



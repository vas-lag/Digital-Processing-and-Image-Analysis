# -*- coding: utf-8 -*-
"""
Created on Sat May  1 03:29:34 2021

@author: Billy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.io
from PIL import Image

def read_image(path):
    mat = scipy.io.loadmat(path)
    return mat

def plot_image(im, title=None):
    plt.figure()
    plt.imshow(im)
    if title:
        plt.title(title)
    
def plot_g(im, title=None):
    plt.figure()
    plt.imshow(im, cmap='gray')
    if title:
        plt.title(title)
        
def plot_hist(hist, title):
    plt.figure()
    if hist.shape[1] > 1:
        plt.plot(np.arange(256), hist[:, 0], color='r')
        plt.plot(np.arange(256), hist[:, 1], color='g')
        plt.plot(np.arange(256), hist[:, 2], color='b')
    else:
        plt.plot(np.arange(256), hist)
    if title:
        plt.title(title)
    
def create_histogram(im):
    hist = np.zeros((256, im.shape[2]))
    for c in range(im.shape[2]):
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                hist[im[x, y, c], c] += 1
    return hist

def update_histogram(hist, rem, add):
    for c in range(rem.shape[2]):
        for x in range(rem.shape[0]):
                hist[rem[x, 0, c], c] -= 1
                hist[add[x, 0, c], c] += 1
    return hist

def create_correspondance(im, hist):
    correspondance = np.zeros((256, im.shape[2]))
    for c in range(im.shape[2]):
        for j in range(256):
            s = 0
            for i in range(j):
                s += hist[i, c] 
            correspondance[j, c] = 255 * s / (im.shape[0] * im.shape[1])
    return correspondance.astype(np.uint8)

def calibrate(im, corr):
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            for c in range(im.shape[2]):
                im[x, y, c] = corr[im[x, y, c], c]
    return im

def equalize(im, plot=False):
    hist = create_histogram(im)
    if plot:
        plot_hist(hist, 'histogram of original image')
    correspondance = create_correspondance(im, hist)
    new_image = calibrate(im, correspondance)
    if plot:
        new_hist = create_histogram(new_image)
        plot_hist(new_hist, 'histogram of equalized image')
    return new_image

def create_correspondance_part(pixel, hist, shape):
    correspondance = np.zeros((1, shape[2]))
    for c in range(shape[2]):
        s = 0
        for i in range(pixel[c]):
            s += hist[i, c] 
        correspondance[0, c] = 255 * s / (shape[0] * shape[1])
    return correspondance.astype(np.uint8)

def equalize_part(part, hist, mode, prev=None):
    mid = part.shape[0] // 2
    if mode == 0:
        hist = create_histogram(part)
    elif mode == 1:
        hist = update_histogram(hist, np.expand_dims(prev, 1), np.expand_dims(part[:, -1, :], 1))
    return create_correspondance_part(part[mid, mid], hist, part.shape), hist

def adaptive_equalize(im, block):
    pad = block // 2
    im_copy = np.zeros((im.shape[0] + 2 * pad, im.shape[1] + 2 * pad, im.shape[2]), dtype=np.uint8)
    im_copy[pad:-pad, pad:-pad, :] = im
    im_copy = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    new_image = np.zeros(im.shape)
    hist = create_histogram(im_copy[:2 * pad + 1, :2 * pad + 1])
    for x in range(im.shape[0]):
        new_image[x, 0], hist = equalize_part(im_copy[x:x + 2 * pad + 1, :2 * pad + 1], hist, 0)
        for y in range(1, im.shape[1]):
            new_image[x, y], hist = equalize_part(im_copy[x:x + 2 * pad + 1, y:y + 2 * pad + 1], hist, 1, im_copy[x:x + 2 * pad + 1, y - 1])
    return new_image.astype(np.uint8)

def equalize_hsi(im_hsi):
    i = np.expand_dims(im_hsi[:, :, 2].astype(np.uint8), 2)
    i_equalized = equalize(i, False) 
    im_hsi_equalized = np.copy(im_hsi)
    im_hsi_equalized[:, :, 2] = i_equalized.squeeze()
    return im_hsi_equalized


def hsi_adaptive_equalize(im_hsi, block):
    i = np.expand_dims(im_hsi[:, :, 2].astype(np.uint8), 2)
    i_adaptive_equalized = adaptive_equalize(i, block) 
    im_hsi_adaptive_equalized = np.copy(im_hsi)
    im_hsi_adaptive_equalized[:, :, 2] = i_adaptive_equalized.squeeze()
    return im_hsi_adaptive_equalized



mat = read_image('../images/5/barbara.mat')
im = mat["barbara"]
# plot_image(im, 'original image')
# im_equalized = equalize(im, True) 
# plot_image(im_equalized, 'image after histograme equalization')

# im_adaptive_equalized = adaptive_equalize(im, 61)
# plot_image(im_adaptive_equalized, 'image after adaptive equalization')


# im_hsi = matplotlib.colors.rgb_to_hsv(im)
# im_hsi_equalized = equalize_hsi(im_hsi)
# im_equalized = matplotlib.colors.hsv_to_rgb(im_hsi_equalized).astype(np.uint8)
# plot_image(im_equalized, 'image equalized on hsi color space')

# im_hsi_adaptive_equalized = hsi_adaptive_equalize(im_hsi, 51)
# im_adaptive_equalized = matplotlib.colors.hsv_to_rgb(im_hsi_adaptive_equalized).astype(np.uint8)
# plot_image(im_adaptive_equalized, 'image after adaptive equaization on hsi color space')


mat = read_image('../images/5/circle.mat')
im = mat["circle"]
im = (im * 255).astype(np.uint8)
plot_g(im, 'original image')
im_equalized = equalize(np.expand_dims(im, 2), True) 
plot_g(im_equalized.squeeze(), 'image contents revealed')




    
    
    
    
    
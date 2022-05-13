# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:16:00 2021

@author: Billy
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_image(path):
    im = Image.open(path)
    im = np.array(im)
    return im

def apply_mask(im, mask):
    res = np.zeros((im.shape[0], im.shape[1], 1))
    im_padded = np.pad(im, ((1, 1), (1, 1), (0, 0)))
    for x in range(res.shape[0]):
        for y in range(res.shape[1]):
            res[x, y] = np.sum(im_padded[x:x+3, y:y+3, 0] * mask)
    res = abs(res)
    maxim = np.max(res)
    return (res * 255 / maxim).astype(np.uint8)

def apply_threshold(im, threshold):
    for x in range(im. shape[0]):
        for y in range(im.shape[1]):
            if im[x, y] >= threshold:
                im[x, y] = 1
            else:
                im[x, y] = 0
                
def create_histogram(im):
    hist = np.zeros((256))
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            hist[im[x, y]] += 1
    return hist

def find_threshold(im):
    hist = create_histogram(im)
    sb = np.zeros((256))
    for t in range(1, 255):
        omega0 = np.sum(hist[:t])
        omega1 = np.sum(hist[t:])
        helper0 = np.arange(t)
        helper1  =np.arange(t, 256)
        m0 = np.sum(helper0 * hist[:t]) / omega0
        m1 = np.sum(helper1 * hist[t:]) / omega1
        sb[t] = omega0 * omega1 * ((m0 - m1) ** 2)
    return np.argmax(sb), hist

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
    

im = read_image('../images/6/factory.jpg')
plot_image(im, 'original image')
sobel_mask_horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_mask_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
masked_horizontal = apply_mask(im, sobel_mask_horizontal)
plot_g(masked_horizontal, 'horizontal sobel mask')
masked_vertical = apply_mask(im, sobel_mask_vertical)
plot_g(masked_vertical, 'vertical sobel mask')

threshold_h, hist_h = find_threshold(masked_horizontal)
plt.figure()
plt.plot(np.arange(256), hist_h)
apply_threshold(masked_horizontal, threshold_h)
plot_g(masked_horizontal, 'horizontal sobel mask after threshold')

threshold_v, hist_v = find_threshold(masked_vertical)
plt.figure()
plt.plot(np.arange(256), hist_v)
apply_threshold(masked_vertical, threshold_v)
plot_g(masked_vertical, 'vertical sobel mask after threshold')
# -*- coding: utf-8 -*-
"""
Created on Sat May  1 00:30:42 2021

@author: Billy
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
import scipy.signal as ss
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
    return im.astype(np.uint8), 255 * var

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
    return np.abs(im)

def P(im):
    N = im.shape[0]
    abs_im = np.abs(im)
    return abs_im ** 2 / N ** 2

def calculate_wiener(im, var, mode=0):
    dft = fft_row_col(im)
    Pg = P(dft)
    if mode == 0:
        Pn = var * 5
    else:
        mid = im.shape[0]
        Pn = np.mean(np.abs(dft)[mid-10:mid+10, mid-10:mid+10]) / 255
        print(Pn)
    Pf = Pg - Pn
    return Pf / (Pf + Pn)

def apply_wiener(im, Hw):    
    fft = fft_row_col(im)
    return inverse_fft_row_col(Hw * fft)
    
       
im = read_image('../images/4/chart.tiff')
plot_g(im, 'original image')
im_gaussian, var = add_gaussian_noise_snr(im, 10)
plot_g(im_gaussian, 'image with gaussian noise')

Hw = calculate_wiener(im_gaussian, var, 1)
im_restored = apply_wiener(im_gaussian, Hw)
plot_g(im_restored, 'restored image')
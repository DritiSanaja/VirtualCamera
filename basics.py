# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
from numba import njit # conda install numba
import numpy as np
from scipy import stats

@njit
def histogram_figure_numba(np_img):
    height, width, _ = np_img.shape
    r_hist = np.zeros(256, dtype=np.float32)
    g_hist = np.zeros(256, dtype=np.float32)
    b_hist = np.zeros(256, dtype=np.float32)

    for i in range(height):
        for j in range(width):
            r = np_img[i, j, 0]
            g = np_img[i, j, 1]
            b = np_img[i, j, 2]
            r_hist[r] += 1.0
            g_hist[g] += 1.0
            b_hist[b] += 1.0

    max_val = max(r_hist.max(), g_hist.max(), b_hist.max())
    if max_val > 0:
        for i in range(256):
            r_hist[i] = (r_hist[i] / max_val) * 3.0
            g_hist[i] = (g_hist[i] / max_val) * 3.0
            b_hist[i] = (b_hist[i] / max_val) * 3.0

    return r_hist, g_hist, b_hist





####

### All other basic functions
def image_statistics(np_img):
    stats_dict = {}
    for i, color in enumerate(['R', 'G', 'B']):
        channel = np_img[:, :, i]
        stats_dict[color] = {
            'mean': np.mean(channel),
            'mode': int(stats.mode(channel.flatten(), axis=None)[0]),
            'std': np.std(channel),
            'max': np.max(channel),
            'min': np.min(channel)
        }
    return stats_dict
####

#entropy calculation
def channel_entropy(channel):
    hist = np.histogram(channel.flatten(), bins=256, range=[0, 256])[0]
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def image_entropy(np_img):
    return {c: channel_entropy(np_img[:, :, i]) for i, c in enumerate(['R', 'G', 'B'])}


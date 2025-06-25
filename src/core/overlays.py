# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 13:18:55 2021

@author: droes
"""

import numpy as np
import cv2 # conda install opencv
from matplotlib import pyplot as plt # conda install matplotlib


# For students
def initialize_hist_figure():
    '''
    Usually called only once to initialize the hist figure.
    Do not change the essentials of this function to keep the performance advantages.
    https://www.youtube.com/watch?v=_NNYI8VbFyY
    '''
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    ax.set_xlim([-0.5, 255.5])
    # fixed size (you can normalize your values between 0, 3 or other ranges to never exceed this limit)
    ax.set_ylim([0,3])
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    def_x_line = np.arange(0, 256, 1)
    # def_y_line = np.zeros(shape=(256,))
    r_plot = ax.plot(def_x_line, def_x_line, 'r', animated=True)[0]
    g_plot = ax.plot(def_x_line, def_x_line, 'g', animated=True)[0]
    b_plot = ax.plot(def_x_line, def_x_line, 'b', animated=True)[0]
    
    return fig, ax, background, r_plot, g_plot, b_plot



def update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars):
    '''
    Uses the initialized figure to update it accordingly to the new values.
    Do not change the essentials of this function to keep the performance advantages.
    '''
    fig.canvas.restore_region(background)        
    r_plot.set_ydata(r_bars)        
    g_plot.set_ydata(g_bars)        
    b_plot.set_ydata(b_bars)

    ax.draw_artist(r_plot)
    ax.draw_artist(g_plot)
    ax.draw_artist(b_plot)
    fig.canvas.blit(ax.bbox)
    
    

def plot_overlay_to_image(np_img, plt_figure):
    '''
    Use this function to create an image overlay.
    You must use a matplotlib figure object.
    Please consider to keep the figure object always outside code loops (performance hint).
    Use this function for example to plot the histogram on top of your image.
    White pixels are ignored (transparency effect)-
    Do not change the essentials of this function to keep the performance advantages.
    '''
    
    rgba_buf = plt_figure.canvas.buffer_rgba()
    (w, h) = plt_figure.canvas.get_width_height()
    imga = np.frombuffer(rgba_buf, dtype=np.uint8).reshape(h,w,4)[:,:,:3]
    
    # ignore white pixels
    plt_indices = np.argwhere(imga < 255)

    # add only non-white values
    height_indices = plt_indices[:,0]
    width_indices = plt_indices[:,1]
    
    np_img[height_indices, width_indices] = imga[height_indices, width_indices]

    return np_img



def plot_strings_to_image(np_img, list_of_string, text_color=(255,0,0), right_space=400, top_space=50):
    '''
    Plots the string parameters below each other, starting from top right.
    Use this function for example to plot the default image characteristics.
    Do not change the essentials of this function to keep the performance advantages.
    '''
    y_start = top_space
    min_size = right_space
    line_height = 20
    (h, w, c) = np_img.shape
    if w < min_size:
        raise Exception('Image too small in width to print additional text.')
        
    if h < top_space + line_height:
        raise Exception('Image too small in height to print additional text.')
    
    y_pos = y_start
    x_pos = w - min_size

    for text in list_of_string:
        if y_pos >= h:
            break
        # SLOW!
        np_img = cv2.putText(cv2.UMat(np_img), text, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        y_pos += line_height

    if type(np_img) is cv2.UMat:
        np_img = np_img.get()

    return np_img

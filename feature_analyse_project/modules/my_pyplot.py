#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:45:55 2018

@author: DD
"""

import matplotlib.pyplot as plt

def display_image_in_actual_size(np_img, dpi = 300, scale = 1):
    """Create and show figure to show a np_img in actual size if dpi is
    chosen correct.
    Almost directly from some stackoverflow comments.

    Arguments:
        np_img: a numpy array with shape (height, width, channels)

        dpi: actually ppi. pixels per inch.

    Returns:
        None
    """
    height = np_img.shape[0]
    width = np_img.shape[1]
    #What size does the figure need to be in inches to fit the image?
    figsize = width * scale / float(dpi), height * scale / float(dpi)
    #Create a figure of the right size with one axes that takes up
    #the full figure
    fig = plt.figure(figsize = figsize, dpi = dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    #Hide spines, ticks, etc.
    ax.axis('off')
    #Display the image.
    ax.imshow(np_img)
    plt.margins(0,0)
    plt.show()
    #TODO Not sure if closing has any advantages
    #see: https://matplotlib.org/tutorials/introductory/pyplot.html
#    If you are making lots of figures, you need to be aware of one more thing:
#      the memory required for a figure is not completely released until the
#      figure is explicitly closed with close(). Deleting all references to
#      the figure, and/or using the window manager to kill the window in which
#      the figure appears on the screen, is not enough, because pyplot maintains
#      internal references until close() is called.
    plt.clf()

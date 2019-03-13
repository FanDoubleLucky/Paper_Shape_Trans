# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:30:06 2019
基于griddata的变形
@author: FYZ
#"""

import os
import sys
import numpy as np
from scipy.interpolate import griddata
from PIL import Image
import cv2

# Read in image and convert to greyscale array object
img_name = '30.jpg'
im = Image.open(img_name)
im = im.convert('RGBA')
im = im.rotate(0, expand=1)
im = np.array(im)
# A meshgrid of pixel coordinates
nx, ny = im.shape[1], im.shape[0]
X1, Y1 = np.meshgrid(np.arange(0, nx, 1, dtype=np.float32), np.arange(0, ny, 1, dtype=np.float32))
for i in range(int(nx*0.0)+1, nx):
    X1[:, i] = X1[:, i-1]+2

# for i in range(int(nx*0.6), nx):
#     Y1[:, i] = Y1[:, i]+i-int(nx*0.6)
# ix = np.random.randint(im.shape[1], size=nsamples)
# iy = np.random.randint(im.shape[0], size=nsamples)

Y = np.arange(0, ny, 1)
for i in range(1, nx):
    Y = np.append(Y, np.arange(0, ny, 1))
X = np.zeros(ny, dtype=np.int16)
for i in range(1, nx):
    X = np.append(X, np.zeros(ny, dtype=np.int16)+i)
samples = im[Y, X]
# int_im = griddata((Y, X), samples, (Y1, X1), method='cubic')
int_im = cv2.remap(im, X1, Y1, interpolation=cv2.INTER_CUBIC)
out = Image.fromarray(np.uint8(int_im))
rot = out.rotate(-0, expand=1)
rot.save('out.PNG')

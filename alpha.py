# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:53:18 2019
各种测试
@author: FYZ
"""

from PIL import Image
import numpy as np
import cv2
from scipy import misc
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
import math

def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * (np.sin(4*np.pi*y**2)**2)



img = Image.open('30.jpg')
img = img.convert('RGBA')
im = np.array(img)
ix, iy = im.shape[1], im.shape[0]
X, Y = np.meshgrid(np.arange(0, ix, 1, dtype=np.float32), np.arange(0, iy, 1, dtype=np.float32))
#
# cv2.getPerspectiveTransform()
# perspective = cv2.warpPerspective(im, M, (ix, iy), cv2.INTER_LINEAR)

p = 50
for i in range(int(0.5*iy), 0, -1):
    X[i, :] = X[i, :] - math.pow(2*p*(int(0.6*iy)-i), 0.5) + math.pow(2*p*(int(0.1*iy)), 0.5)
for i in range(int(0.5*iy), 0, -1):
    Y[i, :] = Y[i+1, :] - 1.25

image = np.zeros(im.shape, im.dtype)
image = cv2.remap(im, X, Y, interpolation=cv2.INTER_CUBIC)
out = Image.fromarray(image)
out = out.rotate(0, expand=1)
out.save('out.png')
#image = Image.open('tsj.jpg').convert('RGBA')
#im1 = image.rotate(32,expand = 1)
#
#im1.show()
#im1.save('out.png')
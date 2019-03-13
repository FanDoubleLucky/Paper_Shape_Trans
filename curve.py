# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:53:18 2019

@author: FYZ
"""

import os
import sys
import numpy as np
from scipy.interpolate import griddata
from PIL import Image
import math
import datetime


def fold_tilt(horizontal, vecrtical, angle, location, image, fold_value):
    '''
    倾斜方向上折纸，如折角
    :param horizontal: 1, right; 0, left
    :param vecrtical: 1, up; 0, down
    :param angle: int, the abs of rotate angle
    :param location: the point where the folding begins, percent
    :param image:
    :param fold_value: folding level
    :return: image folded
    '''
    if vecrtical == 0:
        angle = -1*angle
    else:
        angle = angle
    image = image.rotate(angle, expand = 1)
    im_np = np.array(image)
    nx, ny = im_np.shape[1], im_np.shape[0]
    X1, Y1 = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))

    if horizontal == 1:
        for i in range(int(nx * location), nx):
            X1[:, i] = X1[:, i - 1] + fold_value
    else:
        for i in range(int(nx * location), 0, -1):
            X1[:, i] = X1[:, i + 1] - fold_value

    Y = np.arange(0, ny, 1)
    for i in range(1, nx):
        Y = np.append(Y, np.arange(0, ny, 1))
    X = np.zeros(ny, dtype=np.int16)
    for i in range(1, nx):
        X = np.append(X, np.zeros(ny, dtype=np.int16) + i)

    samples = im_np[Y, X]
    int_im = griddata((Y, X), samples, (Y1, X1), method='cubic')
    image = Image.fromarray(np.uint8(int_im))
    image = image.rotate(-1*angle, expand=1)
    return image


def paper_trigcurve(direction, curve_frequency, curve_amp, location, phase, image):
    '''
    水平垂直方向上三角函数形状的形变，wave
    :param direction: 0，向右；1，向左，2，向下，3，向上
    :param curve_frequency:
    :param curve_amp:
    :param location: percent, the begin location of curving
    :param phase:
    :param image:
    :return:
    '''
    im_np = np.array(image)
    nx, ny = im_np.shape[1], im_np.shape[0]
    X1, Y1 = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))

    if direction == 0:
        for i in range(int(location*nx), nx):
            Y1[:, i] = Y1[:, i] + curve_amp*math.sin(curve_frequency*i+phase)
    elif direction == 1:
        for i in range(nx, int(location*nx), -1):
            Y1[:, i] = Y1[:, i] + curve_amp*math.sin(curve_frequency*(int(location*nx)-i)+phase)
    elif direction == 2:
        for i in range(int(location*ny), ny):
            X1[:, i] = X1[:, i] + curve_amp * math.sin(curve_frequency * (i - int(location * ny)) + phase)
    elif direction == 3:
        for i in range(int(location*ny), 0, -1):
            X1[:, i] = X1[:, i] + curve_amp * math.sin(curve_frequency * (int(location * ny) - i) + phase)

    Y = np.arange(0, ny, 1)
    for i in range(1, nx):
        Y = np.append(Y, np.arange(0, ny, 1))
    X = np.zeros(ny, dtype=np.uint8)
    for i in range(1, nx):
        X = np.append(X, np.zeros(ny, dtype=np.uint8) + i)

    samples = im_np[Y, X]
    int_im = griddata((Y, X), samples, (Y1, X1), method='cubic')
    image = Image.fromarray(np.uint8(int_im))
    image = image.rotate(-1*angle, expand=1)
    return image


def paper_rollcurve(vert_impres_up, vert_impres_down, focal_dis_up, focal_dis_down, location, image):
    '''
    将纸张卷折
    :param vert_impres_up: 纸张上半部分的翻折，上半部分垂直方向上压缩程度，值越大越严重，值的数量级：1, 2左右（1815*2420图像下）
    :param vert_impres_down: 纸张下半部分的翻折，上半部分垂直方向上压缩程度
    :param focal_dis_up: 纸张上半部分的水平方向上的偏移，偏移量符合为抛物线方程，此值为抛物线的焦距，数量级：600左右（1815*2420图像下），抛物线向上开口，此值越小弯曲越严重，从开始到顶端如（900， 899， 898...）偏移量增速从慢到快，数值递增
    :param focal_dis_down: 纸张上半部分的水平方向上的偏移
    :param location: 翻折纸张的中心线位置，为百分比小数
    :param image:
    :return:
    '''
    im_np = np.array(image)
    nx, ny = im_np.shape[1], im_np.shape[0] # nx is the width of image, ny is the height
    X1, Y1 = np.meshgrid(np.arange(0, nx, 1, dtype=np.float16), np.arange(0, ny, 1, dtype=np.float16))

    for i in range(int(location*ny), -1, -1):
        X1[i, :] = X1[i, :] - (0.5/focal_dis_up)*math.pow(int(location*ny)-i, 2)
    for i in range(int((location+0.02)*ny), ny, 1):
        X1[i, :] = X1[i, :] - (0.5/focal_dis_down)*math.pow(i-int((location+0.02)*ny), 2)

    for i in range(int(location*ny), -1, -1):
        Y1[i, :] = Y1[i+1, :] - 1 - vert_impres_up*( (1.0*int(location*ny)-i) / int(location*ny) )
    for i in range(int((location+0.02)*ny), ny, 1):
        Y1[i, :] = Y1[i-1, :] + 1 + vert_impres_down*( 1.0*(i-int((location+0.02)*ny)) / (1.0*ny-int((location+0.05)*ny)) )

    Y = np.arange(0, ny, 1, dtype=np.int16)
    for i in range(1, nx):
        Y = np.append(Y, np.arange(0, ny, 1, dtype=np.int16))
    X = np.zeros(ny, dtype=np.int16)
    for i in range(1, nx):
        X = np.append(X, np.zeros(ny, dtype=np.int16) + i)

    samples = im_np[Y, X]
    int_im = griddata((Y, X), samples, (Y1, X1), method='cubic')
    image = Image.fromarray(np.uint8(int_im))
    return image


# Read in image and convert to grayscale array object

angle = 0
img_name = '30.jpg'
im = Image.open(img_name)
im = im.convert('RGBA')
bg_transparent = Image.new(mode='RGBA', size=(int(1.2*im.size[0]), int(1.2*im.size[1])))
bg_transparent.paste(im, (0, 0, im.size[0], im.size[1]))
# im = fold_tilt(0,0,45,0.3,im,2)
im = paper_trigcurve(3, 0.015, 30, 0, im)
start = datetime.datetime.now()
im = paper_rollcurve(1.5, 1, 600, 900, 0.48, bg_transparent)
end = datetime.datetime.now()
print(end-start)
# for i in range(int(ny*0.5), 0, -1):
#     Y1[i, :] = Y1[i+1, :] - 2

im.save('out.PNG')
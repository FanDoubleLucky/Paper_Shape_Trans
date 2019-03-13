# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:53:18 2019
从基于griddata修改成的基于griddata的变形
@author: FYZ
"""

import os
import sys
import numpy as np
from scipy.interpolate import griddata
from PIL import Image
import math
import datetime
import cv2


def open_after_folded(direction, location, droop, image):
    '''
    纸张垂直方向上折叠后打开，一半的纸张会因为折叠有一些翘起，因为重力平面会下垂形成曲面，用抛物线的形状模拟这种曲面
    :param direction: 0，上半部分；1，下半部分
    :param location: 折叠线位置，百分数，参数建议量：0.4-0.6
    :param droop: 下垂程度，抛物线的焦距，值越小越平，建议量：10-60
    :param image
    :return:
    '''

    im_np = np.array(image)
    nx, ny = im_np.shape[1], im_np.shape[0]
    X1, Y1 = np.meshgrid(np.arange(0, nx, 1, dtype=np.float32), np.arange(0, ny, 1, dtype=np.float32))

    if direction == 0:
        for i in range(int(location * ny), -1, -1):
            X1[i, :] = X1[i, :] - math.pow(2 * droop * (int((location+0.1) * ny) - i), 0.5) + math.pow(2 * droop * (int(0.1 * ny)), 0.5)  # 此处direction+0.1的原因是为了跳过抛物线一开始的陡峭阶段，为了保证逐行偏移量是从0开始变化，要填补跳过的0.1的部分
        vertical_bias = 1.2  # 垂直方向上缩短后与原图片的百分比，1.25时为1/1.25
        for i in range(int(location * ny), -1, -1):
            Y1[i, :] = Y1[i + 1, :] - vertical_bias
    else:
        for i in range(int(location * ny), ny):
            X1[i, :] = X1[i, :] - math.pow(2 * droop * (i - int(location * ny) + 0.1*ny), 0.5) + math.pow(2 * droop * 0.1 * ny, 0.5)  # 此处direction+0.1的原因是为了跳过抛物线一开始的陡峭阶段，为了保证逐行偏移量是从0开始变化，要填补跳过的0.1的部分
        vertical_bias = 1.1  # 垂直方向上缩短后与原图片的百分比，1.25时为1/1.25
        for i in range(int(location * ny), ny):
            Y1[i, :] = Y1[i - 1, :] + vertical_bias

    int_im = cv2.remap(im_np, X1, Y1, interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(np.uint8(int_im))
    return image


def fold_tilt(horizontal, vertical, angle, location, image, fold_value):
    '''
    倾斜方向上折纸，如折角
    :param horizontal: 1, right; 0, left
    :param vertical: 1, up; 0, down
    :param angle: int, the abs of rotate angle
    :param location: the point where the folding begins, percent
    :param image:
    :param fold_value: folding level
    :return: image folded
    '''
    if vertical == 0:
        angle = -1*angle
    else:
        angle = angle
    image = image.rotate(angle, expand=0)
    im_np = np.array(image)
    nx, ny = im_np.shape[1], im_np.shape[0]
    X1, Y1 = np.meshgrid(np.arange(0, nx, 1, dtype=np.float32), np.arange(0, ny, 1, dtype=np.float32))

    if horizontal == 1:
        for i in range(int(nx * location), nx):
            X1[:, i] = X1[:, i - 1] + fold_value
    else:
        for i in range(int(nx * location), 0, -1):
            X1[:, i] = X1[:, i + 1] - fold_value

    int_im = cv2.remap(im_np, X1, Y1, interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(np.uint8(int_im))
    image = image.rotate(-1*angle, expand=0)
    return image


def paper_trigcurve(direction, curve_frequency, curve_amp, location, phase, image):
    '''
    水平垂直方向上三角函数形状的形变，wave
    :param direction: 0，向右；1，向左，2，向下，3，向上
    :param curve_frequency: 数量级，0.1-0.2左右，越大抖动越强烈
    :param curve_amp:
    :param location: percent, the begin location of curving
    :param phase:
    :param image:
    :return:
    '''
    im_np = np.array(image)
    nx, ny = im_np.shape[1], im_np.shape[0]
    X1, Y1 = np.meshgrid(np.arange(0, nx, 1, dtype=np.float32), np.arange(0, ny, 1, dtype=np.float32))

    if direction == 0:
        for i in range(int(location*nx), nx):
            Y1[:, i] = Y1[:, i] + curve_amp*math.sin(curve_frequency*i+phase)
    elif direction == 1:
        for i in range(int(location*nx), 0, -1):
            Y1[:, i] = Y1[:, i] + curve_amp*math.sin(curve_frequency*(int(location*nx)-i)+phase)
    elif direction == 2:
        for i in range(int(location*ny), ny):
            X1[i, :] = X1[i, :] + curve_amp * math.sin(curve_frequency * (i - int(location * ny)) + phase)
    elif direction == 3:
        for i in range(int(location*ny), 0, -1):
            X1[i, :] = X1[i, :] + curve_amp * math.sin(curve_frequency * (int(location * ny) - i) + phase)

    int_im = cv2.remap(im_np, X1, Y1, interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(np.uint8(int_im))
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
    X1, Y1 = np.meshgrid(np.arange(0, nx, 1, dtype=np.float32), np.arange(0, ny, 1, dtype=np.float32)) #X1 ny行[0, 1, 2..., nx-1]; Y1 nx列[0, 1, 2..., ny-1].T

    for i in range(int(location*ny), -1, -1):
        X1[i, :] = X1[i, :] - (0.5/focal_dis_up)*math.pow(int(location*ny)-i, 2)
    for i in range(int((location+0.02)*ny), ny, 1):
        X1[i, :] = X1[i, :] - (0.5/focal_dis_down)*math.pow(i-int((location+0.02)*ny), 2)

    for i in range(int(location*ny), -1, -1):
        Y1[i, :] = Y1[i+1, :] - 1 - vert_impres_up*( (1.0*int(location*ny)-i) / int(location*ny) )
    for i in range(int((location+0.02)*ny), ny, 1):
        Y1[i, :] = Y1[i-1, :] + 1 + vert_impres_down*( 1.0*(i-int((location+0.02)*ny)) / (1.0*ny-int((location+0.05)*ny)) )

    int_im = cv2.remap(im_np, X1, Y1, interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(np.uint8(int_im))
    return image


# Read in image and convert to grayscale array object

img_name = '30.jpg'
im = Image.open(img_name)
im = im.convert('RGBA')
bg_transparent = Image.new(mode='RGBA', size=(int(1.6*im.size[0]), int(1.6*im.size[1])))
bg_transparent.paste(im, (0+int(0.3*im.size[0]), 0+int(0.3*im.size[0]), im.size[0]+int(0.3*im.size[0]), im.size[1]+int(0.3*im.size[0])))
start = datetime.datetime.now()
bg_transparent = open_after_folded(0, 0.4, 20, bg_transparent)
bg_transparent = open_after_folded(1, 0.4, 5, bg_transparent)

bg_transparent = fold_tilt(0, 0, 45, 0.3, bg_transparent, 2)

bg_transparent = fold_tilt(1, 0, -15, 0.3, bg_transparent, 1.4)

# bg_transparent = fold_tilt(0, 0, 25, 0.4, bg_transparent, 2)
bg_transparent = paper_trigcurve(3, 0.005, 30, 0.7, 0, bg_transparent)


# bg_transparent = paper_rollcurve(1.5, 1, 600, 900, 0.48, bg_transparent)
end = datetime.datetime.now()
print(end-start)
# for i in range(int(ny*0.5), 0, -1):
#     Y1[i, :] = Y1[i+1, :] - 2

bg_transparent.save('out.PNG')
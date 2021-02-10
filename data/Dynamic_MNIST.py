#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import cv2
from math import *
import math
from scipy import signal

def mv_mnist(num_sample, seeds, mnist):
    data = np.zeros((num_sample, 10, 128, 128))
    for i in range(num_sample):
        np.random.seed(seeds[i])
        da = np.zeros((10, 256, 256))
        da2 = np.zeros((10, 256, 256))
        x2 = np.random.randint(64, 192-28)
        y2 = np.random.randint(64, 192-28)
        x_start = np.random.randint(80, 160)
        y_start = np.random.randint(80, 160)
        x_0v = np.random.randint(-3, 3)
        y_0v = np.random.randint(-3, 3)
        x_a = int(np.random.normal()*0.1)
        y_a = int(np.random.normal()*0.1)
        x, y = x_start, y_start
        ang_0 = np.random.randint(-6, 6)
        ang_a = int(np.random.normal()*0.1)     
        angles = 0
        da3 = np.zeros((10, 128, 128))
        x3 = np.random.randint(0, 100)
        y3 = np.random.randint(0, 100)
        mask = np.random.rand(1).reshape(1, 1) 
        mask /= np.sum(mask) if np.sum(mask)!=0 else mask
        mask = (1 + np.random.uniform(-0.5, 0.5)) * mask
        da3[0, y3:y3+28, x3:x3+28] = mnist[seeds[i]+20000]
        for t in range(10):
            ang = ang_0 + ang_a*t
            angles += ang
            trans = cv2.getRotationMatrix2D((20, 20), angles, 1)
            mn = np.zeros((40, 40))
            mn[6:34, 6:34] = mnist[seeds[i]]
            da2[t, x2:x2+40, y2:y2+40]  = cv2.warpAffine(mn, trans, (40, 40))
            x_v = x_0v + x_a*t
            y_v = x_0v + y_a*t 
            x += x_v
            y += y_v
            try:
                da[t, x:x+28, y:y+28] = mnist[seeds[i]+30000]
            except:
                da = np.zeros_like(da)
                print(i, t, x_start, y_start)
                break
            if t>0:
                da_t = signal.convolve2d(da3[t-1], mask, mode = 'same')
            else:
                da_t = signal.convolve2d(da3[t], mask, mode = 'same')                
            f = np.fft.fft2(da_t)
            fshift = np.fft.fftshift(f)
            fshift[128-128//4:128+128//4, 128-128//4:128+128//4] = 0
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            da3[t] = img_back
        da3 = da3/(da3.max())*255
        data[i] = da[:, 64:192, 64:192] + da2[:, 64:192, 64:192] + da3
    return data      


def main():
    mnist = np.load('mnist.npz')
    index = np.arange(0, 2000, 1)[::8]
    train_data = mv_mnist(250, index, mnist['X'].reshape(60000, 28, 28))
    index = np.arange(2000, 3000, 1)[::]
    valid_data = mv_mnist(1000, index, mnist['X'].reshape(60000, 28, 28))
    index = np.arange(3000, 6000, 1)[::]  
    test_data = mv_mnist(3000, index, mnist['X'].reshape(60000, 28, 28))
    np.save('./train_data', train_data)
    np.save('./valid_data', valid_data)  
    np.save('./test_data', test_data)
    print('save all data')

if __name__ == '__main__':
    main()


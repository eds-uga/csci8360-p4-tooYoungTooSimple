# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 23:05:07 2016

@author: colin
"""
import os
import json
import matplotlib.pyplot as plt
from numpy import array, zeros
from scipy.misc import imread
from glob import glob

path = '/home/colin/Documents/CSCI8360/Project4/neurofinder.00.00/images'
num = 512 ** 2
f1 = open("00.00Pixel", "w")
for dirname, dirnames, filenames in os.walk(path):
    for filename in sorted(filenames):
        if filename.endswith('.tiff'):
           #data.append(filename)
           files = sorted(glob('/home/colin/Documents/CSCI8360/Project4/neurofinder.00.00/images/'+str(filename)))
           imgs = [imread(f) for f in files]
           tmp = sum(imgs[0].tolist(), []) # convert array to list
           f1.write(str(tmp) + "\n")


f1.close()



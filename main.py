# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import glob
import sys
import os
from models import *
import yaml


blur_images = np.load("./Flower_Images.npy")
blur_images = blur_images[:57]
norm_images = np.load("./Norm_Flower_Images.npy")
print "Data Loaded"

blur_images = 1/127.0*(blur_images-127.0)
norm_images = 1/127.0*(norm_images-127.0)

# print blur_images[5], norm_images[5]

log_dir = "./logs/"
model_path = log_dir+sys.argv[1]

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(model_path+"/results")
    os.makedirs(model_path+"/tf_graph")
    os.makedirs(model_path+"/saved_model")


Unet = UNET(model_path)
Unet.build_model()
Unet.train_model(inputs = [norm_images, blur_images])


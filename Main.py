# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import numpy as np
import glob
import sys
import os
from Models import *


images = np.load("./DataSets/cifar_training.npy")
print "Data Loaded"
images = 1/255.0*images
images = images.reshape((images.shape[0],3,32,32)).transpose(0,2,3,1)

log_dir = "./logs/"
model_path = log_dir+sys.argv[1]

if not os.path.exists(model_path):
    os.makedirs(model_path)
    os.makedirs(model_path+"/results")
    os.makedirs(model_path+"/tf_graph")
    os.makedirs(model_path+"/saved_model")

cifar_gan = GAN(model_path)
cifar_gan.Build_Model()
cifar_gan.Train_Model(inputs = images)


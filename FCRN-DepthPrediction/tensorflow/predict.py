# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import models
import glob
from scipy.misc import imsave

def predict(model_data_path, image_path):

    print image_path
    # Default input size
    height = 256
    width = 256
    channels = 3
    batch_size = 1
   
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)
        
    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')

        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        #net.load(model_data_path, sess) 

        # Evalute the network for the given image
        for image in image_path:
              # Read image
            img = Image.open(image)
            img = img.resize([width,height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis = 0)
            pred = sess.run(net.get_output(), feed_dict={input_node: img})
            imsave("../../pix2pix-tensorflow/data/depth/"+image[37:], pred[0,:,:,0])
            print str(image[37:])+" saved.." 
        # Plot result
        # fig = plt.figure()
        # ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        # fig.colorbar(ii)
        # plt.show()
        
        # return pred
        
                
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help='Converted parameters for the model')
    parser.add_argument('image_paths', help='Directory of images to predict')
    args = parser.parse_args()

    image_path = glob.glob("../../pix2pix-tensorflow/data/normal/*.jpg")
    predict(args.model_path, image_path)

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import cv2
import glob

def create_npy():
	images_list = glob.glob("../pix2pix-tensorflow/data/depth/*.jpg") 
	
	print images_list

	final_size = 256
	image_list = []
	print len(images_list)
	for image in images_list:
		img = cv2.imread(image)
		w = img.shape[1]
		h = img.shape[0]
		ar = float(w)/float(h)
		if w<h:
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,a/2:a/2+final_size ]
		else:
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img
		image_list.append(final_image)
		cv2.imwrite("../pix2pix-tensorflow/data/depth/"+image[33:], final_image)

		print w, h , final_image.shape,image

	# images_array = np.array(image_list)
	# print len(images_array)
	# np.save("HumanTrainingBlur.npy", images_array)

def test_npy():
	# images = np.load("Flower_Images.npy")
	images = np.load("HumanTrainingBlur.npy")
	i=1
	j=1
	print len(images)
	for image in images:
		cv2.imshow("Window", image)
		k=cv2.waitKey(0)
		if k==115:
			cv2.imwrite("./some/"+str(i)+".jpg", image)
			i+=1
		print j
		j+=1


create_npy()
# test_npy()
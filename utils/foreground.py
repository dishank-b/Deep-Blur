# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import cv2
import glob

def gen_image(image):
	img = cv2.imread(image)
	mask = cv2.imread("../pix2pix-tensorflow/data/blurred/"+image[34:-3]+"png")

	result = img.copy()

	# print mask[:,:]==[0,0,0]
	result[mask==[0,0,0]] = 0

	cv2.imshow("window",result)
	cv2.waitKey(0)
	# cv2.imwrite("../pix2pix-tensorflow/data/foreground/"+image[34:], result)
	print image[34:]+ " done"

image_path = glob.glob("../pix2pix-tensorflow/data/normal/*.jpg")
for image in image_path:
	gen_image(image)
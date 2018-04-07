import cv2
import numpy as np
import glob


image_path = glob.glob("../pix2pix-tensorflow/data/depth/*.jpg")
for image in image_path:
	img = cv2.imread(image,0)
	print img.shape
	# cv2.imwrite("../pix2pix-tensorflow/data/depth/"+image[33:], img)
	# print image[33:]+" done"
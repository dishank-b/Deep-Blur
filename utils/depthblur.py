# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import cv2
import glob

def gen_image(image):
	img = cv2.imread(image)
	mask = cv2.imread("../pix2pix-tensorflow/data/blurred/"+image[34:-3]+"png")
	dep_map = cv2.imread("../pix2pix-tensorflow/data/depth/"+image[34:], 0)
	dep_map = cv2.resize(dep_map,dsize=(img.shape[0], img.shape[1]))

	result = img.copy()

	mini=10000
	maxi=0

	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if np.array_equal(mask[i,j],[0,0,0]):
				# img[i,j] = temp[i,j]
				# print i, j
				if(dep_map[i,j]<mini):
					mini = dep_map[i,j]
					# print mini, i,j
				if dep_map[i,j]>maxi:
					maxi = dep_map[i,j]
					# print maxi, i,j

	a = (maxi-mini)/5
	lis = [mini, mini+a, mini+2*a, mini+3*a, mini+4*a,maxi]

	l = [[],[],[],[],[]]
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if np.array_equal(mask[i,j],[0,0,0]):
				# print i,j
				if ((dep_map[i,j]>=lis[0]) and (dep_map[i,j]<lis[1])):
					l[0].append([i,j])
				elif ((dep_map[i,j]>=lis[1]) and (dep_map[i,j]<lis[2])):
					l[1].append([i,j])
				elif ((dep_map[i,j]>=lis[2]) and (dep_map[i,j]<lis[3])):
					l[2].append([i,j])
				elif ((dep_map[i,j]>=lis[3]) and (dep_map[i,j]<lis[4])):
					l[3].append([i,j])
				elif ((dep_map[i,j]>=lis[4]) and (dep_map[i,j]<lis[5])):
					l[4].append([i,j])


	m = 0
	check_list = []
	for i in xrange(5,35,6):
		temp = img.copy()
		temp_an = cv2.GaussianBlur(temp,(i,i),0)
		# cv2.imshow("cow", temp)
		# cv2.waitKey(0) 
		for a in l[m]:
			result[a[0], a[1]] = temp_an[a[0], a[1]]
		m = m+1

	cv2.imwrite("../pix2pix-tensorflow/data/blur/"+image[34:], result)
	print image[30:]+ " done"

image_path = glob.glob("../pix2pix-tensorflow/data/normal/*.jpg")
for image in image_path:
	gen_image(image)
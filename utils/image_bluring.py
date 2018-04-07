import numpy as np
import cv2
import glob

masks = glob.glob("./tmep/blurred/*.png")
images = glob.glob("./tmep/normal/*.jpg")

masks.sort(key=lambda f: int(filter(str.isdigit, f)))
images.sort(key=lambda f: int(filter(str.isdigit, f)))

k = 1
for image, mask in zip(images, masks):
	img = cv2.imread(image)
	bin_mask  = cv2.imread(mask)
	temp = img
	temp = cv2.GaussianBlur(temp,(15,15),0)
	for i in range(bin_mask.shape[0]):
		for j in range(bin_mask.shape[1]):
			if bin_mask[i,j].any(0):
				temp[i,j] = img[i,j]
	cv2.imwrite("./tmep/training_imgs/"+str(k)+".jpg", temp)
	k+=1
import numpy as np
import cv2
import glob

masks = glob.glob("../dataset/training_imgs/*.jpg")
images = glob.glob("../dataset/normal/*.jpg")

masks.sort(key=lambda f: int(filter(str.isdigit, f)))
images.sort(key=lambda f: int(filter(str.isdigit, f)))

k=1
for image, mask in zip(images, masks):
	img = cv2.imread(image,0)
	bin_mask  = cv2.imread(mask,0)
	cv2.imwrite("../dataset/gray/normal/"+str(k)+".jpg", img)
	cv2.imwrite("../dataset/gray/training_imgs/"+str(k)+".jpg", bin_mask)
	k+=1
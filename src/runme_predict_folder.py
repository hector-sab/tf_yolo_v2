import os
import cv2
import numpy as np
from tqdm import tqdm
from model import Model

import utils as ut
import constants as ct

im_dir = '/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/416_416/00000001/'
out_dir = '/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/416_416/00000001_predicted/'

if __name__=='__main__':
	model = Model()

	imfpaths = sorted(os.listdir(im_dir))

	for i in tqdm(range(len(imfpaths))):
		imfpath = imfpaths[i]
		#print(imfpath)
		if imfpath[-3:]=='jpg':
			#print('YES')
			im = ut.load_im(im_dir+imfpath)
			coord,lbs,ob_mask,pobj = model.predict(im)
			filtered_objects = ut.nms_yolo(coord,lbs,ob_mask,pobj)
			#ut.prediction2kitti(coord,lbs,ob_mask,out_dir,imfpath[:-3]+'txt')
			ut.prediction2kitti2(filtered_objects,out_dir,[imfpath[:-3]+'txt'])
# Calculates how many bboxes were correctly predicted and
# how many were not correctly predicted, or if any gt bbox 
# was not predicted

import os
from glob import glob
import numpy as np

import utils as ut


if __name__=='__main__':
	# Indicate directories of interest
	gt_dir = '/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/416_416/00000001/'
	pred_dir = '/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/416_416/00000001_predicted/'

	# Analyze the iou with different th
	out = ut.compare_kitti_files_folder(gt_dir,pred_dir)
	for clss in out.keys():
		print(clss,out[clss])
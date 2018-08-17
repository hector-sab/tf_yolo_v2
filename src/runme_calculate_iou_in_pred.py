import os
from glob import glob
import numpy as np

from utils import Rectangle,IoU,Detections

def check_valid_files(files,fmt='txt'):
	# Checks that only the files with the specified format
	# are returned
	# Args:
	#   files (list): Contains the path of the files of interest
	#   fmt (str): What's the format extension of interest

	vfiles = []
	for fpath in files:
		if fpath[-1*len(fmt):]==fmt:

			vfiles.append(fpath)
	return(vfiles)

if __name__=='__main__':
	# Indicate directories of interest
	gt_dir = '/data/Documents/CICATA/Test/yolo/CICATA_dataset/416_416/00000001/'
	pred_dir = '/data/Documents/CICATA/Test/yolo/CICATA_dataset/416_416/00000001_predicted/'


	# Load the file paths
	gt_files = sorted(os.listdir(gt_dir))
	pred_files = sorted(os.listdir(pred_dir))
	
	gt_files = check_valid_files(gt_files)
	pred_files = check_valid_files(pred_files)

	folders = {'gt':gt_files,'pred':pred_files}

	if len(folders['gt'])>len(folders['pred']):
		bigger = 'gt'
		smaller = 'pred'
	else:
		bigger = 'pred'
		smaller = 'gt'

	list_of_analysis = [] # List of lists. Each iner list contains
	                      # a list of lists containing the best
	                      # iou and the objecets compared per object
	                      # of each one of the images
	                      
	# Iterate over the folder with more files
	for i in range(len(folders[bigger])):
		fname = folders[bigger][i]
		list_of_ious = [] # list of list. Each inner list contains
		                  # [0]: Best Intersection Over Union
		                  # [1]: (gt ind obj, pred ind obj) 
		if fname in folders[smaller]:
			gt = Detections(fpath=gt_dir+fname)
			pred = Detections(fpath=pred_dir+fname)

			print('{:03d} - GT: {} PR: {}'.format(i,len(gt.objects),len(pred.objects)))

			# Determine the IoU of each object
			for i,gt_obj in enumerate(gt.objects):
				best_iou = None
				best_iou_ind = None
				for j,pr_obj in enumerate(pred.objects):
					if gt_obj.clss==pr_obj.clss:
						iou = IoU(gt_obj.bbox,pr_obj.bbox)
						if best_iou is None and iou is not None:
							best_iou = iou
							best_iou_ind = (i,j)
						elif iou<best_iou:
							best_iou = iou
							best_iou_ind = (i,j)

				if best_iou is not None and best_iou<0.5:
					best_iou = None
					best_iou_ind = None

				list_of_ious.append([best_iou,best_iou_ind])

		list_of_analysis.append(list_of_ious)

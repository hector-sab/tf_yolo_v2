import os
import pickle
import numpy as np
from tqdm import tqdm
from model import Model

import utils as ut

im_dir = '/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/416_416/00000001/'
out_dir = '/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/'
out_dir += '416_416/{}/00000001_predicted_pTH-{}_sTH-{}/'

if __name__=='__main__':
	# Test name
	t_name = 'test_01'
	model = Model()

	# Loads the images paths
	imfpaths = sorted(os.listdir(im_dir))
	imfpaths = ut.check_valid_files(imfpaths,'jpg')

	# Controls the prob th in the prediction model
	pTH = np.linspace(0,1,21)

	# Controls minimum iou th for the nms
	sTH = np.linspace(0,1,21)

	pbar = tqdm(range(pTH.shape[0]*sTH.shape[0]*len(imfpaths)))
	# Predict all the bboxes
	# Iterates over all the probs THs
	for p in pTH:
		# Iterates over all the ious THs
		for s in sTH:
			final_out_dir = out_dir.format(t_name,p,s)
			if not os.path.isdir(final_out_dir):
				os.makedirs(final_out_dir)
			for i in range(len(imfpaths)):
				imfpath = imfpaths[i]
				im = ut.load_im(im_dir+imfpath,new_shape=(416,416))
				coord,lbs,ob_mask,pobj = model.predict(im,TH_prob=p)
				filtered_objects = ut.nms_yolo(coord,lbs,ob_mask,pobj,TH=s)
				ut.prediction2kitti2(filtered_objects,final_out_dir,[imfpath[:-3]+'txt'])
				pbar.update(1)
	pbar.close()

	# Calculate performance
	gt_dir = im_dir

	# results contains all the results in the following format:
	# {'p-0.5_s-0.5': __results__}
	results = {}

	pbar = tqdm(range(pTH.shape[0]*sTH.shape[0]))
	# Predict all the bboxes
	# Iterates over all the probs THs
	for p in pTH:
		# Iterates over all the ious THs
		for s in sTH:
			pred_dir = out_dir.format(t_name,p,s)
			out = ut.compare_kitti_files_folder(gt_dir,pred_dir)
			results['p-{:.2f}_s-{:.2f}'.format(p,s)] = out
			pbar.update(1)
	pbar.close()

	with open('full_results.pk','w') as f:
		pickle.dump(results,f)
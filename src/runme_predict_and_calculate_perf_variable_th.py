import os
import pickle
import numpy as np
from tqdm import tqdm
from model import Model

import utils as ut

main_dir = '/data/HectorSanchez/respaldo/Documents/CICATA/Test/yolo/CICATA_dataset/416_416/'
im_dir = main_dir+'00000001/'
out_dir = main_dir+'{}/00000001_predicted_pTH-{:.2f}_sTH-{:.2f}/'

# Test name
t_name = 'test_01'

if __name__=='__main__':
	# Controls the prob th in the prediction model
	pTH = np.linspace(0,1,21)

	# Controls minimum iou th for the nms
	sTH = np.linspace(0,1,21)

	if False:
		model = Model()

		# Loads the images paths
		imfpaths = sorted(os.listdir(im_dir))
		imfpaths = ut.check_valid_files(imfpaths,'jpg')


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

	with open(os.path.join(main_dir,t_name,'full_results.pk'),'wb') as f:
		pickle.dump(results,f)
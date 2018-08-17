import os
import cv2
import numpy as np
from tqdm import tqdm
from model import Model,OBJECTS

im_dir = '/data/Documents/CICATA/Test/yolo/CICATA_dataset/416_416/00000001/'
out_dir ='/data/Documents/CICATA/Test/yolo/CICATA_dataset/416_416/00000001_predicted/'

def load_image(path):
	im = cv2.imread(path)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	im = (im / 255.).astype(np.float32)
	im = np.expand_dims(im, 0)
	return(im)

def prediction2kitti(coords,lbs,ob_mask,dest,fname):
	num_pred = coords.shape[0]

	for i in range(num_pred):
		bmask = ob_mask[i] # Boolean mask  indicating which anchor has an object
		cmask = np.sum(ob_mask[i],axis=-1) # Cell mask indicating which cell contains an obj 
			                              # at any of its anchors

		lines = []
		#print('--->',cmask.shape)
		# Iterate over all cells 
		for ii,crow in enumerate(cmask):
			for jj,cell in enumerate(crow):
				if cell>0:
					# At this point there's at least one object
					# detected in this cell
					for kk,anc in enumerate(bmask[ii,jj]):
						if anc:
							# [y_max,x_max,y_min,x_min]
							bbox = coord[i,ii,jj,kk]
							lbl = lbs[i,ii,jj,kk]

							# [x_min,y_min,x_max,y_max]
							line = '{0} 0.0 0 0.0 {1} {2} {3} {4} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n'
							line = line.format(OBJECTS[lbl],int(bbox[1]),int(bbox[0]),int(bbox[3]),int(bbox[2]))
							lines.append(line)

		# Save lines into a file
		with open(dest+fname,'w+') as f:
			for line in lines:
				f.write(line)

if __name__=='__main__':
	model = Model()

	imfpaths = sorted(os.listdir(im_dir))

	for i in tqdm(range(len(imfpaths))):
		imfpath = imfpaths[i]
		#print(imfpath)
		if imfpath[-3:]=='jpg':
			#print('YES')
			im = load_image(im_dir+imfpath)
			coord,lbs,ob_mask,pobj = model.predict(im)
			prediction2kitti(coord,lbs,ob_mask,out_dir,imfpath[:-3]+'txt')
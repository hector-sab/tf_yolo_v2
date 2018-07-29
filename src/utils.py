import cv2
import numpy as np
import matplotlib.pyplot as plt

def txt2list(path):
	"""
	Reads the 'path' file and returns all the lines
	as a list of strings.
	"""
	with open(path,'r') as f:
		content = f.readlines()
	lst = [x.strip('\n') for x in content]
	return(lst)

def draw_output(ims,coord,lbs,ob_mask,pobj=None):
	"""
	Draws the bboxes detected in the images
	Args:
		im (np.array): It should contain a batch of images to be 
				drawn. Shape [?,im_h,im_w,im_c]
		coord (np.array): It should contain the coordinates of the
				detected objects. Shape [?,GRID_H,GRID_W,NUM_ANCHORS,4].
				'4' stands for [by1,bx1,by2,bx2]
		lbs (np.array): It should contain the label of the bounding box detected.
				Shape [?,GRID_H,GRID_W,NUM_ANCHORS]
		ob_mask (np.array): It should contain the indicator if an element in the grid
				actually contains an object or not.
		pobj (None | np.array): If different than None, it indicates the probability of
				each object of being that object. Shape [?,GRID_H,GRID_W,NUM_ANCHORS]
	"""
	num_ims = ims.shape[0]
	ims_out = np.zeros_like(ims)

	for i,im in enumerate(ims):
		bmask = ob_mask[i] # Boolean mask  indicating which anchor has an object
		cmask = np.sum(bmask,axis=-1) # Cell mask indicating which cell contains an obj 
		                              # at any of its anchors

		for ii,crow in enumerate(cmask):
			for jj,cell in enumerate(crow):
				# Iterates over all the cells in cmask
				if cell>0:
					# There's an object detected in any of the anchors of this class
					#print('wuuuu---:',ii,cell)

					for kk,anc in enumerate(bmask[ii,jj]):
						if anc:
							#print('\t{0},{1},{2},{3} Yes man'.format(i,ii,jj,kk))
							bbox = coord[i,ii,jj,kk]
							#print('\tCoord: ',bbox)
							lbl = lbs[i,ii,jj,kk]
							#print('\tLabel: ',lbl)

							im = cv2.rectangle(im,(bbox[1],bbox[0]),(bbox[3],bbox[2]),color=(255, 0, 0), thickness=3)

		ims_out[i] = im
	return(ims_out)

def load_im(path,im_h=416,im_w=416):
	"""
	Loads the image to be used in the YOLO v2 model.

	It returns a np.array with the shape [1,im_h,im_w,im_c]

	Args:
		path (str): Indicates where to find the image
		im_h (int): Height of the output image
		im_w (int): Width of the output image
	"""
	img = cv2.imread(path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (im_h, im_w))
	img = (img / 255.).astype(np.float32)
	img = np.expand_dims(img, 0)
	return(img)

def plot_ims(ims):
	"""
	Plot all the images in the batch

	Args: 
		ims (np.array): Is an array containing a batch of images.
			Its shape is [?,im_h,im_w,im_c]
	"""
	for im in ims:
		plt.imshow(im)
		plt.show()
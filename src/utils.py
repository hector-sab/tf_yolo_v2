import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def iou(bbox1,bbox2):
	"""
	Args:
		bbox (tf.Tensor): Shape [?,13,13,5,4]
						[x,y,w,h]
	"""
	A_area = tf.multiply(bbox1[...,2],bbox1[...,3])
	B_area = tf.multiply(bbox2[...,2],bbox2[...,3])

	bbox1 = pred2coord(bbox1[...,0],bbox1[...,1],bbox1[...,2],bbox1[...,3])
	bbox2 = pred2coord(bbox2[...,0],bbox2[...,1],bbox2[...,2],bbox2[...,3])
	"""
	bbox (tf.Tensor): Shape [?,13,13,5,4]
					[y_top,x_left,y_botom,x_right]
	"""
	# Here the reference frame changes. Y axis becomes positive for the upper
	# part. Normally with images the upper part "is" negative
	A_max = tf.stack([bbox1[...,2],bbox1[...,3]],axis=-1) # Upper Right
	A_min = tf.stack([bbox1[...,0],bbox1[...,1]],axis=-1) # Lower Left

	B_max = tf.stack([bbox2[...,2],bbox2[...,3]],axis=-1) # Upper Right
	B_min = tf.stack([bbox2[...,0],bbox2[...,1]],axis=-1) # Lower Left

	inters_max = tf.minimum(A_max,B_max) # Upper Right
	inters_min = tf.maximum(A_min,B_min) # Lower Left

	# Check if they indeed have an union
	hw_inters = tf.maximum(inters_max-inters_min,tf.constant(0.))
	
	inters_area = tf.multiply(hw_inters[...,0],hw_inters[...,1])
	area_union = tf.subtract(tf.add(A_area,B_area),inters_area)
	epsilon = 1e-30
	iou = tf.divide(inters_area,tf.add(area_union,tf.constant(epsilon)))

	return(iou)

def predC2grid(x,y,grid_w=13,grid_h=13):
	"""
	Converts from predicted centroid in cell to grid space
	Args:
		x,y (tf.Tensor): Tensor containing the values
			to be converted with the shape	[?,13,13,5]
		grid_h|w (int): Indicates the shape of the grid
	"""
	# Lets get the bx and by
	coord_x = tf.range(grid_w)
	coord_y = tf.range(grid_h)

	Cx,Cy = tf.meshgrid(coord_x,coord_y)
	Cx = tf.cast(Cx,tf.float32)
	Cy = tf.cast(Cy,tf.float32)

	## In here we are going to condition the Cx,Cy so instead of being of
	## shape [13,13] it will add an extra dimension at the end so it can be
	## broadcasted with x,y and summed up happily
	Cx = tf.reshape(Cx,[grid_h,grid_w,1])
	Cy = tf.reshape(Cy,[grid_h,grid_w,1])

	Gx = tf.add(x,Cx)
	Gy = tf.add(y,Cy)

	return(Gx,Gy)

def grid2norm(x,y,grid_w=13,grid_h=13):
	"""
	Converts grid space into normalized space [0,1]
	Args:
		x,y (tf.Tensor): Tensor containing the values
			to be normalized with the shape	[?,13,13,5]
		grid_h|w (int): Indicates the shape of the grid
	"""
	Nx = tf.div(x,tf.constant(grid_w,dtype=tf.float32))
	Ny = tf.div(y,tf.constant(grid_h,dtype=tf.float32))

	return(Nx,Ny)

def norm2imspace(x,y,im_w=416,im_h=416):
	"""
	Converts from norm space to image space 
	Args:
		x,y (tf.Tensor): Tensor containing the values
			to be normalized with the shape	[?,13,13,5]
		im_h|w (int): Indicates the shape of the image
	"""
	Ix = tf.multiply(x,tf.constant(im_w,dtype=tf.float32))
	Iy = tf.multiply(y,tf.constant(im_w,dtype=tf.float32))

	return(Ix,Iy)

def predS2grid(w,h,anchors):
	"""
	Converts from norm space to image space 
	Args:
		w,h (tf.Tensor): Tensor containing the values
			to be normalized with the shape	[?,13,13,5]
		anchors (np.array): Contains the shape of the anchors
	"""
	Pw = tf.cast(tf.constant(anchors[:,1]),tf.float32,name='Pw')
	Gw = tf.multiply(w,Pw)

	Ph = tf.cast(tf.constant(anchors[:,0]),tf.float32,name='Ph')
	Gh = tf.multiply(h,Ph)

	return(Gw,Gh)

def pred2coord(x,y,w,h):
	"""
	Converts from centroid/shape to 2-points coordinates.
	Inputs:
		x,y,w,h (tf.Tensor): Contains the location of the bounding box
			as in form of the centroid (xy) and its shape (w,h). Shape
			of each one of them: [?,13,13,5]
	Output:
		coord (tf.Tensor): Returns the 2-points coordinates of the bounding
			box containing the object. Shape [?,13,13,5,4] where '4' is 
			[y_top,x_left,y_bottom,x_right]
	"""
	centroid = tf.concat([tf.expand_dims(y,axis=-1),tf.expand_dims(x,axis=-1)],axis=-1)
	shape = tf.concat([tf.expand_dims(h,axis=-1),tf.expand_dims(w,axis=-1)],axis=-1)

	bbox_min = centroid - (shape/2)
	bbox_max = centroid + (shape/2)

	coord = tf.concat([bbox_min[...,0:1], bbox_min[...,1:2], bbox_max[...,0:1], bbox_max[...,1:2]], axis=-1)

	return(coord)

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
				'4' stands for [by1,bx1,by2,bx2]. Top left, bottom rights
		lbs (np.array): It should contain the label of the bounding box detected.
				Shape [?,GRID_H,GRID_W,NUM_ANCHORS]
		ob_mask (np.array): It should contain the indicator if an element in the grid
				actually contains an object or not.
		pobj (None | np.array): If different than None, it indicates the probability of
				each object of being that object. Shape [?,GRID_H,GRID_W,NUM_ANCHORS]

	Note: Everything should be denormalized at this point.
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
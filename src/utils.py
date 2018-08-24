import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import constants as ct

def tf_iou(bbox1,bbox2):
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

def draw_output2(ims,batch):
	# Draws the bboxes detected in the images
	# Args:
	#   im (np.array): It should contain a batch of images to be 
	#       drawn. Shape [?,im_h,im_w,im_c]
	#   batch (list): Contain all the bboxes and probs of the batch
	#       each elemenet of the list is a dict with two keys: 
	#       'bboxes', and 'probs'
	#
	#       'bboxes' (np.array): Contain the coords of the objects.
	#               The shape of the array is [?,4]. 
	#               [Top,Left,Bottom,Right]
	#        'probs' (np.array): Contain the probs of each object.
	#               The shape of the array is [?]
	num_elems = ims.shape[0]
	ims_out = np.zeros_like(ims)

	for i in range(num_elems):
		im = ims[i]
		classes = batch[i].keys()
		for clss in classes:
			num_bboxes = batch[i][clss]['bboxes'].shape[0]
			for j in range(num_bboxes):
				bbox = batch[i][clss]['bboxes'][j]
				prob = batch[i][clss]['probs'][j]

				im = cv2.rectangle(im,(bbox[1],bbox[0]),(bbox[3],bbox[2]),color=(255, 0, 0), thickness=3)

		ims_out[i] = im
	return(ims_out)

def load_im(path,new_shape=None):
	"""
	Loads the image to be used in the YOLO v2 model.

	It returns a np.array with the shape [1,im_h,im_w,im_c]

	Args:
		path (str): Indicates where to find the image
		new_shape (int tuple): If the image should be 
		   reshaped, indicate it here as (height,width)
	"""
	im = cv2.imread(path)
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	if new_shape is not None:
		im = cv2.resize(im, (new_shape[0], new_shape[1]))
	im = (im / 255.).astype(np.float32)
	im = np.expand_dims(im, 0)
	return(im)

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
							line = line.format(ct.OBJECTS[lbl],int(bbox[1]),int(bbox[0]),int(bbox[3]),int(bbox[2]))
							lines.append(line)

		# Save lines into a file
		with open(dest+fname,'w+') as f:
			for line in lines:
				f.write(line)

def prediction2kitti2(batch,dest,fnames):
	# Converts the predictions to a kitti file
	# Args:
	#   batch (list): Each element is a dictionary containing
	#      the 'bboxes' for each class object
	#      I.E = [{15:{'bboxes':np.array([[int,int,int,int]])},
	#              16:{'bboxes':np.array([[int,int,int,int]])}}]
	#
	#        where 15,16 are the classes they represent
	#
	#   dest (str): Destitation folder of the file
	#   fnames (list): Contains all the names of the files

	num_elems = len(batch)

	# Iterate over all the elements of the batch
	for i in range(num_elems):
		lines = [] # Lines of text to be saved in the file
		# Iterate over all the classes in the element
		for clss in batch[i].keys():
			# Iterate over all the objects of the class
			for j in range(batch[i][clss]['bboxes'].shape[0]):
				bbox = batch[i][clss]['bboxes'][j]
				line = '{0} 0.0 0 0.0 {1} {2} {3} {4} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n'
				line = line.format(ct.OBJECTS[clss],int(bbox[1]),int(bbox[0]),int(bbox[3]),int(bbox[2]))
				lines.append(line)

		# Save lines into a file if it contains something
		if len(lines)>0:
			with open(dest+fnames[i],'w+') as f:
				for line in lines:
					f.write(line)




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





######


class Rectangle:
	def __init__(self,centroid=None,shape=None,pt1=None,pt2=None):
		# Define a rectangle given the centroid or 
		# two points.It uses a ordinary cartesian frame.
		#
		# Args:
		#   centoroid (tuple|np.array): (x,y)
		#   shape (tuple|np.array): (w,h)
		#   pt1 (tuple|np.array): (x,y) 
		#   pt2 (tuple|np.array): (x,y)
		if centroid is not None and shape is not None:
			self.centroid = np.array(centroid)
			self.shape = np.array(shape)
			
			self.top_right = self.centroid + self.shape/2
			self.bottom_left = self.centroid - self.shape/2
			self.top_left = np.array([self.bottom_left[0],self.top_right[1]])
			self.bottom_right = np.array([self.top_right[0],self.bottom_left[1]])
		elif pt1 is not None and pt2 is not None:
			###
			pt1 = np.array(pt1)
			pt2 = np.array(pt2)
			# Determine top left and bottom right
			if pt1[0]>pt2[0]:
				# If pt1 is in the right, change it to the left
				tmp = pt1[0]
				pt1[0] = pt2[0]
				pt2[0] = tmp

			if pt1[1]<pt2[1]:
				# If pt1 is in the bottom, change it to the top
				tmp = pt1[1]
				pt1[1] = pt2[1]
				pt2[1] = tmp
			###
			self.top_left = np.array(pt1)
			self.bottom_right = np.array(pt2)
			self.top_right = np.array([self.bottom_right[0],self.top_left[1]])
			self.bottom_left = np.array([self.top_left[0],self.bottom_right[1]])

			self.centroid = ((self.bottom_right + self.top_left)/2).astype(np.int32)
			self.shape = self.top_right - self.bottom_left
		self.area = self.shape[0]*self.shape[1]

def IoU(A,B,rect=False):
	# Calculates the intersection over union of two rectangles
	#
	# Args:
	#   A|B (Rectangle): Rectangles to be compared
	#   rect (bool): Returns the iou value  and a rectangle 
	#      that represents the IoU if True. Else, it 
	#      returns the iou value.

	# Calculate points of inner intersection
	top_right = np.minimum(A.top_right,B.top_right)
	bottom_left = np.maximum(A.bottom_left,B.bottom_left)

	# Calculate shape of Intersection [W,H]
	shape = np.maximum(top_right-bottom_left,0)
	# Calculate area of Intersection
	area = shape[0]*shape[1]

	area_union = A.area + B.area - area
	iou = area / area_union

	# Define the rectangle of the IoU
	if iou>0:
		C = Rectangle(pt1=top_right,pt2=bottom_left)
	else:
		C = None

	if rect:
		return(iou,C)
	else:
		return(iou)

def IoU2(coordA,coordB):
	# Calculates the intersection over union of two different
	# sets of coord
	#   Args:
	#     coordA (np.array): Coord of each object. Shape [?,4]
	#        [Top,Left,Bottom,Right]
	#     coordB (np.array): Same as coordA
	#
	# It returns a matrix I with the IoUs of shape MxN, where M is the number of coords
	# in coordA and N is the number of coords in coordB. 
	# So, the row r in I represents the coord r in coordA, and the col c in I represents
	# the coord c in coordB.
	#
	# A_{} :-> coordA_{} .... B_{} :-> coordB_{}
	#
	# coordA = A_{0} [[a00, a01, a02, a03],    coordB = B_{0}  [[b00, b01, b02, b03],
	#          A_{1} [a10, a11, a12, a13],              B_{1}   [b10, b11, b12, b13]]
	#          A_{2} [a20, a21, a22, a23]]
	#
	#                    B_{0}   B_{1}
	# I =       A_{0}  [[iou00, iou01],
	#           A_{1}   [iou10, iou11],
	#           A_{2}   [iou20, iou21]]  

	if coordA.shape[0]==0 or coordB.shape[0]==0:
		# If there's no objects to compare
		iou = None
	else:
		# Calculate the area of each box in each set
		areaA = (coordA[:,3]-coordA[:,1])*(coordA[:,2]-coordA[:,0])
		areaB = (coordB[:,3]-coordB[:,1])*(coordB[:,2]-coordB[:,0])

		# Determine the IoU of each object
		# Horizontal axis represent the predicted bboxes
		# Vertical axis represents the gt bboxes

		# Chooses the top (in image coord frame) position among all
		top = np.maximum(coordA[:,0:1],coordB[:,0])
		# Chooses the right position among all
		right = np.minimum(coordA[:,3:4],coordB[:,3])
		# Chooses the bottom (in image coord frame) position among all
		bottom = np.minimum(coordA[:,2:3],coordB[:,2])
		# Chooses the left position among all
		left = np.maximum(coordA[:,1:2],coordB[:,1])

		# Calculate the shape
		# Calculate the height
		height = np.maximum(bottom - top,0)
		# Calculate the width
		width = np.maximum(right - left,0)

		area_inters = height*width
		area_union = areaA.reshape(-1,1) + areaB - area_inters
		iou = area_inters/area_union

	return(iou)

class Cosa:
	# Contains the information of an objecet.
	# It indicates what type of object is, and
	# where it is located
	def __init__(self,clss,pt1,pt2):
		self.clss = clss
		self.bbox = Rectangle(pt1,pt2)

class Detections:
	# Reads the files with the objects descriptions
	# and stores them in memory in an easy way to access
	def __init__(self,fpath,format='kitty'):
		# Formats ['kitty','xml']
		self.fpath = fpath
		self.objects = self.file2objects()

	def file2objects(self):
		# Read the file
		with open(self.fpath,'r') as f:
			tmp = f.readlines()
		objects = []

		# Process each line
		for line in tmp:
			line = line.strip('\n')
			line = line.split(' ')
			obj = Cosa(clss=line[0],
				pt1=(int(line[4]),int(line[5])),
				pt2=(int(line[6]),int(line[7])))
			objects.append(obj)

		return(objects)

class Detections2:
	# Reads the files with the objects descriptions
	# and stores them in memory in an easy way to access
	def __init__(self,fpath,format='kitty'):
		# Formats ['kitty','xml']
		self.fpath = fpath
		self.objects = self.file2objects()

	def file2objects(self):
		# Read the file
		with open(self.fpath,'r') as f:
			tmp = f.readlines()
		coords = []
		clss = []

		# Process each line
		for line in tmp:
			line = line.strip('\n')
			line = line.split(' ')
			coords.append([int(line[4]),int(line[5]),
				int(line[6]),int(line[7])])
			clss.append(ct.OBJECTS.index(line[0]))

		return({'coord':np.array(coords),'classes':np.array(clss)})



def compare_kitti_files(gt_path,pred_path,TH=0.5):
	# Determines the number of correspondences between two files
	# for each class.
	#
	# Returns a dictionary of dictionaries. The main dict contain the
	# classes analysed, and the inner one of each class contains three
	# fields: 
	#   - 'correspondence': How many correspondences where found
	#   - 'no_correspondence_with_gt': How many ground truth bboxes
	#         didn't find a correspondence
	#   - 'no_correspondence_wiht_pred': How many predicted bboxes didn't
	#         find a correspondence

	# Load the objects of each file
	gt = Detections2(gt_path)
	pred = Detections2(pred_path)

	# Merge all the classes predicted and the ground truth classes
	clss_gt = gt.objects['classes']
	clss_pred = pred.objects['classes']
	all_clss = np.unique(np.hstack([clss_gt,clss_pred]))

	# It stores the results
	results_per_class = {} 
	# Iterate over all classes 
	for clss in all_clss:
		# Get the coord of the gt/pred for the class
		gt_coord = gt.objects['coord'][gt.objects['classes']==clss,:]
		pred_coord = pred.objects['coord'][pred.objects['classes']==clss,:]

		# Calculate IoU and make sure they are valid
		iou = IoU2(gt_coord,pred_coord)
		
		if iou is None:
			# If there is no objects to make a comparision
			# add them directly to its corresponding container
			if gt_coord.shape[0]==0:
				num_gt_with_no_corr = 0
				num_pred_with_no_corr = pred_coord.shape[0]
			elif pred_coord.shape[0]==0:
				num_pred_with_no_corr = 0
				num_gt_with_no_corr = gt_coord.shape[0]

			results_per_class[clss] = {'correspondence':0,
			    'no_correspondence_with_gt':num_gt_with_no_corr,'no_correspondece_with_pred':num_pred_with_no_corr}
		else:
			# Which iou are valid ious for our purpose
			valid_iou = iou>=TH

			# Check how many gt were found, and how many weren't
			gt_contains_correspondanse = np.sum(valid_iou,axis=1)>0

			# Select only the objects that can be valid based on their iou
			same_gt_pred_obj = iou*valid_iou
			# Select only the objects that really have a correspondance with gt and pred
			same_gt_pred_obj = same_gt_pred_obj[gt_contains_correspondanse!=0]
			# Indicates the index of the gt bboxes that have a correspondance
			ind_valid_gt = np.argwhere(gt_contains_correspondanse).reshape(-1)
			# Indicates which pred bbox correspond to which gt bbox
			# The index of the element correspond to the gt element of the ind_valid_gt
			ind_correspondence = np.argmax(same_gt_pred_obj,axis=1)
			# Indicates what's the iou of the correspondences
			iou_correspondence = same_gt_pred_obj[np.arange(same_gt_pred_obj.shape[0]),ind_correspondence]

			# How many gt and pred bboxes where found?
			num_gt_pred_corr = ind_valid_gt.shape[0]
			# How many gt didn't have a correspondence
			num_gt_with_no_corr = np.argwhere(gt_contains_correspondanse==False).reshape(-1)
			num_gt_with_no_corr = num_gt_with_no_corr.shape[0]
			# How many pred didn't have a correspondence
			num_pred_with_no_corr = pred_coord.shape[0] - num_gt_pred_corr


			results_per_class[clss] = {'correspondence':num_gt_pred_corr,
			    'no_correspondence_with_gt':num_gt_with_no_corr,'no_correspondece_with_pred':num_pred_with_no_corr}

	return(results_per_class)

def compare_kitti_files_folder(gt_dir,pred_dir,TH=0.5):
	# Compares all the files in a single folder of gt and pred for each one
	#   Args:
	#     gt_dir (str): path to the folder containing the ground truth
	#     pred_dir (str): path to the folder containing the predictions

	# Load the file paths
	gt_files = sorted(os.listdir(gt_dir))
	pred_files = sorted(os.listdir(pred_dir))

	gt_files = check_valid_files(gt_files)
	pred_files = check_valid_files(pred_files)

	# In here we are going to count how many of each one per class
	# Each class shuld have a dict as follow
	# {'correspondence':0,'no_correspondence_with_gt':0,'no_correspondece_with_pred':0}
	results = {}


	# In here we are going to track the files already found, or not.
	files_found = []
	
	# Compare each file
	for file in gt_files:
		if file in pred_files:
			# If the file exists in both lists
			files_found.append(file)
			out = compare_kitti_files(os.path.join(gt_dir,file),os.path.join(pred_dir,file),TH)
			
			for clss in out.keys():
				if clss not in list(results.keys()):
					results[clss] = {'correspondence':0,'no_correspondence_with_gt':0,'no_correspondece_with_pred':0}
				# Update the count
				results[clss]['correspondence'] += out[clss]['correspondence']
				results[clss]['no_correspondence_with_gt'] += out[clss]['no_correspondence_with_gt']
				results[clss]['no_correspondece_with_pred'] += out[clss]['no_correspondece_with_pred']
		else:
			# If file only found in the gt list
			out = Detections2(os.path.join(gt_dir,file))

			all_clss = np.unique(out.objects['classes'])
			for clss in all_clss:

				if clss not in list(results.keys()):
					results[clss] = {'correspondence':0,'no_correspondence_with_gt':0,'no_correspondece_with_pred':0}

				results[clss]['no_correspondence_with_gt'] += out.objects['coord'][out.objects['classes']==clss,:].shape[0]

	# Remove all the files found in pred
	for file in pred_files:
		if file not in files_found:
			# If file only exists in the pred list
			out = Detections2(os.path.join(pred_dir,file))

			all_clss = np.unique(out.objects['classes'])
			for clss in all_clss:
				if clss not in list(results.keys()):
					results[clss] = {'correspondence':0,'no_correspondence_with_gt':0,'no_correspondece_with_pred':0}

				results[clss]['no_correspondence_with_pred'] += out.objects['coord'][out.objects['classes']==clss,:].shape[0]

	return(results)

def nms(coord,probs,TH=0.8):
	# Non-Maxima Suppression
	#  Args:
	#    coord (np.array): Shape [?,4]
	#      [Top,Left,Bottom,Right]
	#    probs (np.array): Shape: [?]
	#    TH (float): The minimum intersection over union of
	#       the bboxes to be considered possible same objects
	#
	# Returns the coord and probs of the filtered objects
	np.set_printoptions(linewidth=200,precision=2)
	iou = IoU2(coord,coord)
	valid_iou = iou>=TH

	# Checks if the iou of a cell has multiple repetitions in different rows
	inspect_col = np.sum(valid_iou,axis=0)>1
	inspect_row = np.sum(valid_iou,axis=1)>1

	# List of ind of the real objects
	real_objects = []
	# Retrieve the ind of the rows!=0 in the col==True
	rows_already_checked = []
	# Iterate over all rows to assert the bboxes to be inspected
	for i in range(valid_iou.shape[0]):
		#print('->',i)
		if i not in rows_already_checked:
			# There's more than one bbox in the same row
			# Which cols of those rows have repetition?
			ind_col = np.argwhere(valid_iou[i,:])
			rows_same = [] # Which rows should be merged
			rows_same.append(i)
			for j in ind_col:
				# Check if the col j as repetitions
				j = j[0]
				if inspect_col[j]:
					# Check the rows that are repeating in the col
					rows = np.argwhere(valid_iou[:,j]).reshape(-1)
					rows_same.extend(rows.tolist())

			# Clean the list. Remove the repeated ones
			rows_same = np.unique(np.array(rows_same)).tolist()

			# Add it to the list of already explored
			rows_already_checked.extend(rows_same)

			# Merge the rows_same into a signgle row
			current_row = np.sum(valid_iou[rows_same],axis=0).astype(np.bool)

			real_object = np.argmax(probs*current_row)
			real_objects.append(real_object)

			real_prob = probs[np.argmax(probs*current_row)]
	
	real_objects = np.array(real_objects)

	return(coord[real_objects,:],probs[real_objects])

def nms_yolo(coord,lbs,ob_mask,prob,TH=0.8):
	# Non-Maxima Suppression
	# coord : shape [?,13,13,5,4]
	#              [?,Top,Left,Bottom,Right]
	# lbs : shape [?,13,13,5]
	# pobj : shape [?,13,13,5]
	# ob_mask : shape [?,13,13,5]
	# TH (float): Minimum intersection over union

	nms_elements = []
	for elem in range(coord.shape[0]):
		# Select all the coord of all valid cells
		bbox = coord[elem][ob_mask[elem]]
		# Select the labels of all valid cells
		labels = lbs[elem][ob_mask[elem]]
		# Select the prob of all valid cells
		probs = prob[elem][ob_mask[elem]]
		
		# Let's iterate over all possible classes
		nms_bbox = {}
		for clss in np.unique(labels):
			# Obtain the bboxes and probs of objects of the same class
			cbbox = bbox[labels==clss] # Shape: [?,4]
			cprobs = probs[labels==clss] # Shape: [?]

			filt_coord,filt_prob = nms(cbbox,cprobs,TH)
			nms_bbox[clss] = {'bboxes':filt_coord,'probs':filt_prob}

		nms_elements.append(nms_bbox)
	return(nms_elements)

if __name__=='__main__':
	pass
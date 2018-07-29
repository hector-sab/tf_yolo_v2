import math
import warnings

import cv2
import numpy as np
import tensorflow as tf

from utils import txt2list
from constants import ANCHORS

class Data:
	"""
	Retreives the data for training. We are going to use 
	Kitti files for the labels.
	"""
	def __init__(self,ims_paths_f,lbs_paths_f,im_h=416,im_w=416,grid_h=13,grid_w=13):
		"""
		Args:
			ims_paths_f (str): Path to the file containing all the images paths
			lbs_paths_f (str): Path to the file containing all the labels paths
		"""
		self.clss2lbl = {'person':15}
		self.ANCHORS = ANCHORS
		self.NUM_ANCHORS = self.ANCHORS.shape[0]
		self.GRID_H = grid_h
		self.GRID_W = grid_w
		self.IM_H = im_h
		self.IM_W = im_w

		self.ims_paths = txt2list(ims_paths_f)
		self.lbs_paths = txt2list(lbs_paths_f)
		if len(self.ims_paths)!=len(self.lbs_paths):
			msg = 'ims_paths ({0}) has not the same elements than lbs_paths ({1})'.format(
				len(self.ims_paths),len(self.lbs_paths))
			warnings.warn(msg,Warning)
		
		self.total_it = len(self.ims_paths) # Totatl iterations per epoch
		self.current_item = 0

	def next_batch(self,bs=0):
		"""
		Returns the next batch of images and labels.
		Args:
			bs (int): Size of the batch
		"""
		# '6' represent [xc,yc,w,h,pobj,class]
		out_ims = np.zeros((bs,self.IM_H,self.IM_W,3))
		out_lbs = np.zeros((bs,self.GRID_H,self.GRID_W,self.NUM_ANCHORS,6))

		add2current_item = bs
		for it in range(bs):
			ind = self.current_item + it
			
			if ind>= self.total_it:
				add2current_item = bs - it
				self.current_item = 0
				ind = 0
			
			im = self.__load_im(self.ims_paths[ind])
			lbl = self.__load_lbl(self.lbs_paths[ind])

			out_ims[it] = im
			out_lbs[it] = lbl

		self.current_item += add2current_item

		return(out_ims,out_lbs)

	def reset_ind(self):
		self.current_item = 0

	def __load_lbl(self,path):
		"""
		Loads and adjust the labels for each image
		"""
		out = np.zeros((self.GRID_H,self.GRID_W,self.NUM_ANCHORS,6))
		with open(path,'r') as f:
			tmp = f.readlines()
			for line in tmp:
				line = line.strip('\n')
				line = line.split(' ')
				
				clss = self.clss2lbl[line[0]]
				x_min = float(line[4])
				y_min = float(line[5])
				x_max = float(line[6])
				y_max = float(line[7])

				xc,yc = (y_min+y_max)/2.,(x_min+x_max)/2.
				h, w = y_max - y_min, x_max - x_min

				# Normalize
				xc /= self.IM_W
				yc /= self.IM_H
				w /= self.IM_W
				h /= self.IM_H

				# In which grid they are located
				GW_ind = int(math.floor(xc*self.GRID_W))
				GH_ind = int(math.floor(yc*self.GRID_H))

				# In which anchor they are located
				ANCH_ind = self.__select_anchor(np.array([h,w]))
				print(GH_ind,GW_ind,ANCH_ind)
				out[GH_ind,GW_ind,ANCH_ind] = [xc,yc,w,h,1,clss]
			return(out)

	def __select_anchor(self,shape):
		"""
		Args:
			shape (np.array): (height,width)
		"""
		best_iou = 0
		best_iou_ind = -1
		for ind,anchor in enumerate(self.ANCHORS):
			iou = self.__get_iou(shape,anchor)
			if best_iou_ind<0:
				best_iou = iou
				best_iou_ind = ind
			else:
				if best_iou<iou:
					best_iou = iou
					best_iou_ind = ind
		return(best_iou_ind)

	def __get_iou(self,hw1,hw2):
		"""
		Calculates the intersection over union.
		Args:
			hw (np.array): (height,width)
		"""
		# get extremes of both boxes
		hw1_max, hw2_max = hw1/2., hw2/2.
		hw1_min, hw2_min = -hw1_max, -hw2_max

		# get intersection area
		intersection_min = np.maximum(hw1_min, hw2_min)
		intersection_max = np.minimum(hw1_max, hw2_max)
		hw_intersection = np.maximum(intersection_max-intersection_min, 0.)
		area_intersection = hw_intersection[0] * hw_intersection[1]

		# get union area
		area_hw1 = hw1[0] * hw1[1]
		area_hw2 = hw2[0] * hw2[1]
		area_union = area_hw1 + area_hw2 - area_intersection
		
		iou = area_intersection / area_union
		return(iou)

	def __load_im(self,path):
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
		#img = cv2.resize(img, (im_h, im_w))
		img = (img / 255.).astype(np.float32)
		img = np.expand_dims(img, 0)
		return(img)



class Trainer:
	def __init__(self,model,train_set,val_set=None,lr=3e-7):
		self.lr = lr
		self.train_set = train_set

		self.sess = self.model.sess
		self.model = model
		self.inputs = self.model.inputs

		self.labels = tf.placeholder(tf.float32,[None,
			self.model.last_layer().get_shape()[1].value,
			self.model.last_layer().get_shape()[2].value,
			self.model.last_layer().get_shape()[3].value])

		self.loss = self.__loss()
		self.optimizer = self.__optimization()

	def __loss(self):
		#### S: PRED ####
		mask = self.model.bb_ob_mask # Shape [?,13,13,5]

		xy_norm_pred = self.model.xy_norm # Shape [?,13,13,5,2]
		x_pred = xy_norm_pred[...,0]
		y_pred = xy_norm_pred[...,1]

		wh_norm_pred = self.model.wh_norm # Shape [?,13,13,5,2]
		w_pred = wh_norm_pred[...,0]
		h_pred = wh_norm_pred[...,1]

		pobj_pred = self.model.pobj # Shape [?,13,13,5]
		pclass_pred = self.model.pclass # Shape [?,13,13,5,25]
		#### E: PRED ####

		#### S: LBL ####
		xy_lbl = self.labels[...,0:2] # Shape [?,13,13,5,2]
		x_lbl = xy_lbl[...,0]
		y_lbl = xy_lbl[...,1]

		wh_lbl = self.labels[...,2:4] # Shape [?,13,13,5,2]
		w_lbl = wh_lbl[...,0]
		h_lbl = wh_lbl[...,1]

		pobj_lbl = self.labels[...,4] # Shape [?,13,13,5]
		pclass_lbl = tf.one_hot(self.labels[...,5],self.model.NUM_OBJECTS,axis=-1) # Shape [?,13,13,5,25]
		#### E: LBL ####

		A_x = tf.pow(tf.subtract(x_lbl,x_pred),tf.constant(2.))
		A_y = tf.pow(tf.subtract(y_lbl,y_pred),tf.constant(2.))
		A_sum = tf.add(A_x,A_y)
		A_filt = tf.multiply(A_sum,tf.cast(mask,tf.float32))
		A = tf.reduce_sum(A_filt)

		B_w = tf.pow(tf.sqrt(tf.subtract(w_lbl,w_pred)),tf.constant(2.))
		B_h = tf.pow(tf.sqrt(tf.subtract(h_lbl,h_pred)),tf.constant(2.))
		B_sum = tf.add(B_w,B_h)
		B_filt = tf.multiply(B_sum,tf.cast(mask,tf.float32))
		B = tf.reduce_sum(B_filt)

		C_sub = tf.pow(tf.subtract(pobj_lbl,pobj_pred),tf.constant(2.))
		C_filt = tf.multiply(C_sub,tf.cast(mask,tf.float32))
		C = tf.reduce_sum(C_filt)

		D_sub = tf.pow(tf.subtract(pobj_lbl,pobj_pred),tf.constant(2.))
		D_filt = tf.multiply(D_sub,tf.cast(tf.logical_not(mask),tf.float32))
		D = tf.reduce_sum(D_filt)

		#E = tf.nn.softmax_cross_entropy_with_logits_v2(labels=pclass_lbl,logits=pclass_pred)
		E_closs = tf.pow(tf.subtract(pclass_lbl,pclass_pred),tf.constant(2.))
		E_sum = tf.reduce_sum(E_closs,axis=-1)
		E_filt = tf.multiply(E_sum,tf.cast(mask,tf.float32))
		E = tf.reduce_sum(E_filt)

		loss = A + B + C + D + E
		return(loss)

	def __optimization(self):
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
		return(optimizer)

	def optimize(self,n_iter=None,n_epochs=None,bs=0):
		LAST_BATCH = None

		if n_iter:
			for it in range(n_iter):
				ims,labels = self.train_set.next_batch(bs)
				feed_dict = {self.inputs:ims,self.labels:labels}
				self.sess.run(self.optimizer,feed_dict=feed_dict)

		elif n_epochs:
			total_it_ts = self.train_set.total_it
			if total_it_ts%bs==0:
				total_it = total_it_ts//bs
			else:
				batches = total_it = total_it_ts//bs
				total_it = batches + 1

				LAST_BATCH = total_it_ts - batches*bs

			for epoc in range(n_epochs):
				for it in range(total_it):
					if LAST_BATCH is None:
						ims,labels = self.train_set.next_batch(bs)
					else:
						if it<total_it-2:
							ims,labels = self.train_set.next_batch(bs)
						else:
							ims,labels = self.train_set.next_batch(LAST_BATCH)

					feed_dict = {self.inputs:ims,self.labels:labels}
					self.sess.run(self.optimizer,feed_dict=feed_dict)


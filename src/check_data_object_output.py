import cv2
import numpy as np
import matplotlib.pyplot as plt

from train import Data
from constants import ANCHORS
from utils import draw_output,plot_ims

def decode_out(inp,im_h=416,im_w=416,g_h=13,g_w=13):
	xc = inp[...,0] # shape [?,13,13,5]
	yc = inp[...,1] # shape [?,13,13,5]
	w = inp[...,2] # shape [?,13,13,5]
	h = inp[...,3] # shape [?,13,13,5]
	mask = inp[...,4] # shape [?,13,13,5]
	clss = inp[...,5] # shape [?,13,13,5]

	X,Y = np.meshgrid(np.arange(g_w),np.arange(g_h))
	X = np.expand_dims(X,axis=-1)
	Y = np.expand_dims(Y,axis=-1)
	# To grid space
	xc += X*mask
	yc += Y*mask
	# Normalize
	xc /= g_w
	yc /= g_h
	# To image space
	xc *= im_w
	yc *= im_h
	w *= im_w
	h *= im_h

	centroid = np.stack([yc,xc],axis=-1)
	shape = np.stack([h,w],axis=-1)
	
	xy_min = centroid-(shape/2)
	xy_max = centroid+(shape/2)

	bbox = np.stack([xy_min[...,0],xy_min[...,1],xy_max[...,0],xy_max[...,1]],axis=-1).astype(np.int32)
	print(bbox.shape)
	return(bbox,clss,mask)

if __name__=='__main__':
	ims_paths_f = '../test_train/ims.txt'
	lbs_paths_f = '../test_train/lbs.txt'

	data = Data(ims_paths_f,lbs_paths_f)

	ims,lbs = data.next_batch(bs=10)
	# '6' represent [xc,yc,w,h,pobj,class]
	bbox,clss,mask = decode_out(lbs)
	"""
	for i in range(mask.shape[-1]):
		plt.imshow(mask[0,...,i])
		plt.show()
	plt.imshow(ims[0])
	plt.show()
	"""
	dims =draw_output(ims,bbox,clss,mask)
	plot_ims(dims)
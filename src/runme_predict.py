import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from model import Model
import utils as ut

if __name__=='__main__':
	model = Model()

	im_path = '../test_im/im3.jpg'
	print('Im path: ',im_path)

	im = ut.load_im(im_path)
	print('-->',im.shape)
	
	coord,lbs,ob_mask,pobj = model.predict(im)
	print('coord Shape: {}'.format(coord.shape))
	print('lbs Shape: {}'.format(lbs.shape))
	print('ob_mask Shape: {}'.format(ob_mask.shape))

	import numpy as np
	np.save('coord',coord)
	np.save('lbs',lbs)
	np.save('ob_mask',ob_mask)
	np.save('pobj',pobj)

	ut.nms(coord,lbs,ob_mask,pobj)

	ims = ut.draw_output(im,coord,lbs,ob_mask,pobj)
	ut.plot_ims(ims)
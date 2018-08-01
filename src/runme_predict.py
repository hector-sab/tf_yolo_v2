import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from model import Model
from utils import load_im,draw_output,plot_ims

if __name__=='__main__':
	model = Model()

	im_path = '../test_im/im3.jpg'
	print('Im path: ',im_path)

	im = load_im(im_path)
	
	
	coord,lbs,ob_mask,pobj = model.predict(im)
	print('coord Shape: {}'.format(coord.shape))
	print('lbs Shape: {}'.format(lbs.shape))
	print('ob_mask Shape: {}'.format(ob_mask.shape))

	ims = draw_output(im,coord,lbs,ob_mask,pobj)
	plot_ims(ims)
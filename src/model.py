"""
Description: Main model of the architecture Yolo v2.
"""

# TODO: Fix shape of the output prediction.
#       It's not consistent with the original one
# TODO: Complete draw outputs
# TODO: What the heck just happened at bh,bw and not having to normalize?

import numpy as np
import tensorflow as tf


def num_gpus():
	from tensorflow.python.client import device_lib
	local_device_protos = device_lib.list_local_devices()
	gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
	return(len(gpus))


if num_gpus() > 0:
	DEVICE = '/gpu:0'
	print('Using GPU')
else:
	DEVICE = '/cpu:0'
	print('Using CPU')


OBJECTS = ['tvmonitor', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 
          'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train']

class Net:
	"""
	Base class for all the other classes. Not used directly.
	"""
	def conv(self,inputs,filters,kernel_size,name='conv',params={}):
		"""
		Description:
			Creates the convolutions with more options available.
		
		Args:
			inputs (4-D tf.Tensor): A 4D tensor containin the features to be processed.
			filters (int): Number of filters to be created for the convolution.
			kernel_size (int): Size of the kernel to be used in the convolutions
			name (str): Name of the convolution
			params (dict): Dictionary containing extra parameters
				Keys:
					'dropout': 2-element list indicating if dropout will be used and the prob to be 
						to used. I.E. [True,0.8]
					'batchNorm': Boolean indicating if batch normalization will be used. NOTE: When using
						batch normalization, biases should not be used.
					'bias': Boolean indicating if biases will be used.
					'activation': 1-element or 2-elements list indicating which activation is going to
						be used, or not. I.E. [tf.nn.leaky_relu,0.1] or [tf.nn.relu] or [None]
		"""

		########################################
		# Default parameters for the convolution options. Here are indicated all those that are not
		# the primary options...
		default_params = {'dropout':[False,1.], 'batchNorm':False,'bias':True,'activation':[None]}
		# Checks which parameters where indicated to be modified
		diff = set(default_params.keys()) - set(params.keys())

		# Fills the params dictionary with all the default parameters that weren't indicated in params
		for key in diff:
			params[key] = default_params[key]
		########################################

		with tf.variable_scope(name):
			out = tf.layers.conv2d(inputs=inputs,filters=filters,kernel_size=kernel_size,use_bias=params['bias'],
				padding='SAME')

			if params['batchNorm']:
				out = tf.layers.batch_normalization(inputs=out)

			if params['activation'][0]==tf.nn.leaky_relu:
				out = params['activation'][0](out,alpha=params['activation'][1])
			elif params['activation'][0] is not None:
				out = params['activation'][0](out)
			
			if params['dropout'][0]:
				out = tf.layers.dropout(inputs=out,rate=params['dropout'][1])

			return(out)


	def pool(self,inputs,type_='maxpool',psize=2,strides=2,padding="same",name='pool'):
		"""
		Description:
			Creates the pool operation.

		Args:
			inputs (4-D tf.Tensor): Input tensor.
			type_ (str|int): String indicating which pooling operation to use
				'maxpool' <--> 0
				'average' <--> 1
			psize (int): Pool size of the operation
			strides (int): strides of the operation
		"""

		if type(type_)==int:
			if type_==0:
				type_ = 'maxpool'
			elif type_==1:
				type_ = 'average'


		if type_=='maxpool':
			pool = tf.layers.max_pooling2d(inputs=inputs,pool_size=psize,
						strides=strides,padding='same',name=name)
		elif type_=='average':
			pool = tf.layers.average_pooling2d(inputs=inputs,pool_size=psize,
				strides=strides,padding=padding,name=name)
		return(pool)


	def fc(self,inputs,units,params={},name='fc'):
		"""
		Description:
			Creates the fully connected layers with more options available.
		
		Args:
			inputs (4-D tf.Tensor): A 4D tensor containin the features to be processed.
			units (int): Number of outputs.
			name (str): Name of the convolution
			params (dict): Dictionary containing extra parameters
				Keys:
					'dropout': 2-element list indicating if dropout will be used and the prob to be 
						to used. I.E. [True,0.8]
					'bias': Boolean indicating if biases will be used.
					'activation': 1-element or 2-elements list indicating which activation is going to
						be used, or not. I.E. [tf.nn.leaky_relu,0.1] or [tf.nn.relu] or [None]
		"""
		########################################
		# Default parameters for the convolution options. Here are indicated all those that are not
		# the primary options...
		default_params = {'dropout':[False,1.],'batchNorm':False,'bias':True,'activation':[tf.nn.relu]}
		# Checks which parameters where indicated to be modified
		diff = set(default_params.keys()) - set(params.keys())

		# Fills the params dictionary with all the default parameters that weren't indicated in params
		for key in diff:
			params[key] = default_params[key]
		########################################

		with tf.variable_scope(name):
			out = tf.layers.dense(inputs=inputs,units=units,activation=params['activation'][0],use_bias=params['bias'])

			if params['batchNorm']:
				out = tf.layers.batch_normalization(inputs=out)

			if params['activation'][0]==tf.nn.leaky_relu:
				out = params['activation'][0](out,alpha=params['activation'][1])
			elif params['activation'][0] is not None:
				out = params['activation'][0](out)
			
			if params['dropout'][0]:
				out = tf.layers.dropout(inputs=out,rate=params['dropout'][1])

		return(out)


	def last_layer(self):
		"""
		Return the last layer of the network.

		NOTE: Not intended to be used alone in the Net class, but with the child classes.
		"""
		nlayers = len(self.model)
		return(self.model[nlayers-1])



class Model(Net):
	def __init__(self,inputs=None,sess=None,ckpt_dir='../checkpoints/',verbose=False):
		# Predefined Values
		self.NAME = 'model'
		self.verbose = verbose
		self.CKPT_DIR = ckpt_dir

		self.ANCHORS = np.array(
			[[0.09112895, 0.06958421],
			 [0.21102316, 0.16803947],
			 [0.42625895, 0.26609842],
			 [0.25476474, 0.49848],
			 [0.52668947, 0.59138947]])
		self.NUM_ANCHORS = self.ANCHORS.shape[0]
		self.NUM_OBJECTS = 20
		self.MAX_DETECTIONS_PER_IMAGE = 10
		self.THRESHOLD_OUT_PROB = 0.5
		self.THRESHOLD_IOU_NMS = 0.5

		if not sess:
			self.sess = tf.Session()
		else:
			self.sess = sess

		if not inputs:
			self.inputs = tf.placeholder(tf.float32,shape=[None,416,416,3])	
		else:
			self.inputs = inputs

		self.IM_H = self.inputs.get_shape()[1].value
		self.IM_W = self.inputs.get_shape()[2].value

		self.model = {}

		self.__model()
		self.pred_results = None
		self.__post_model()
		self.__tensorboard()
		self.sess.run(tf.global_variables_initializer())
		self.__init_from_ckpt()

	def __model(self):
		with tf.variable_scope(self.NAME):
			params = {'batchNorm':True,'bias':False,'activation':[tf.nn.leaky_relu,0.1]}
			self.model[0] = self.conv(self.inputs,32,3,params=params,name='conv1')
			self.model[1] = self.pool(self.model[0],name='pool1')

			self.model[2] = self.conv(self.model[1],64,3,params=params,name='conv2')
			self.model[3] = self.pool(self.model[2],name='pool2')

			self.model[4] = self.conv(self.model[3],128,3,params=params,name='conv3')

			self.model[5] = self.conv(self.model[4],64,1,params=params,name='conv4')

			self.model[6] = self.conv(self.model[5],128,3,params=params,name='conv5')
			self.model[7] = self.pool(self.model[6],name='pool5')

			self.model[8] = self.conv(self.model[7],256,3,params=params,name='conv6')

			self.model[9] = self.conv(self.model[8],128,1,params=params,name='conv7')

			self.model[10] = self.conv(self.model[9],256,3,params=params,name='conv8')
			self.model[11] = self.pool(self.model[10],name='pool8')

			self.model[12] = self.conv(self.model[11],512,3,params=params,name='conv9')

			self.model[13] = self.conv(self.model[12],256,1,params=params,name='conv10')

			self.model[14] = self.conv(self.model[13],512,3,params=params,name='conv11')

			self.model[15] = self.conv(self.model[14],256,1,params=params,name='conv12')

			self.model[16] = self.conv(self.model[15],512,3,params=params,name='conv13')
			self.model[17] = self.pool(self.model[16],name='pool13')

			self.model[18] = self.conv(self.model[17],1024,3,params=params,name='conv14')

			self.model[19] = self.conv(self.model[18],512,1,params=params,name='conv15')

			self.model[20] = self.conv(self.model[19],1024,3,params=params,name='conv16')

			self.model[21] = self.conv(self.model[20],512,1,params=params,name='conv17')

			self.model[22] = self.conv(self.model[21],1024,3,params=params,name='conv18')

			self.model[23] = self.conv(self.model[22],1024,3,params=params,name='conv19')

			self.model[24] = self.conv(self.model[23],1024,3,params=params,name='conv20')

			self.model[25] = self.conv(self.model[16],64,1,params=params,name='conv21')

			shrink = tf.space_to_depth(self.model[25],block_size=2)
			concat = tf.concat([shrink,self.model[24]],axis=-1)

			self.model[26] = self.conv(concat,1024,3,params=params,name='conv22')

			# Detector Layer Starts Here
			self.model[27] = self.conv(self.model[26],self.NUM_ANCHORS*(4+1+self.NUM_OBJECTS),1,name='conv23')

			self.GRID_H = self.model[27].get_shape()[1].value
			self.GRID_W = self.model[27].get_shape()[2].value

			# Reshape the output
			self.model[28] = tf.reshape(self.model[27],[-1,self.GRID_H,self.GRID_W,self.NUM_ANCHORS,4+1+self.NUM_OBJECTS],name='pred')

			if self.verbose:
				for i in range(len(self.model)):
					print('\t{}'.format(self.model[i]))

	def __init_from_ckpt(self,verbose=False):
		from tensorflow.python import pywrap_tensorflow
		file_name = tf.train.latest_checkpoint(self.CKPT_DIR)#'../checkpoints/tf/ckpt-61'
		reader = pywrap_tensorflow.NewCheckpointReader(file_name)
		var_to_shape_map = reader.get_variable_to_shape_map()
		keys = list(sorted(var_to_shape_map.keys()))
		valid_ckpt_names = []
		for key in keys:
			if (not 'OPTIMIZER'	in key) and (not 'optimizer'	in key):
				split = key.split('/')
				if len(split)>3:
					valid_ckpt_names.append(key)

		tensors = tf.global_variables()

		for tensor in tensors:
			tsplit = tensor.name.split('/')
			tnum = tsplit[1][4:]
			for key in valid_ckpt_names:
				ksplit = key.split('/')

				# Check to which layer it belongs
				knum = ksplit[1][4:]
				if tnum==knum:
					# Checks that they are the same type of layer
					if ksplit[2] in tsplit[-1]:
						if verbose:
							print('Loading {0} to {1}'.format(key,tensor.name))
						value = reader.get_tensor(key)
						tensor.load(value,session=self.sess)

	def __tensorboard(self):
		"""
		Draws the graph into tensorboard
		"""
		self.yolo_writer = tf.summary.FileWriter('../tensorboard/yolov2',self.sess.graph)


	def predict(self,ims):
		"""
		ims (np.array): Contain a batch of images. Shape [?,IM_H,IM_W,IM_C]
		"""
		# Use __post_model2
		# TODO: Fix pobj to return the prob of the detected objects
		coord,lbs,ob_mask,pobj = self.sess.run([self.coord,self.bb_lbs,
			self.bb_ob_mask,self.pobj],feed_dict={self.inputs:ims})

		return(coord,lbs,ob_mask,pobj)

	def __post_model(self):
		pred = self.last_layer()
		
		# Let the elements be [ty,tx,th,tw,to,pc.....] instead of [tx,ty,tw,tx,to,pc.....]
		#pred = tf.concat([pred[...,1::-1], pred[...,3:1:-1], pred[...,4:]], axis=-1)

		# Condition the outputs
		###### Remember the formulation....
		#
		# bx = sig(tx) + Cx   |   by = sig(ty) + Cy
		# bw = Pw * exp(tw)   |   bh = Ph * exp(th)
		# 
		# Where 
		#   (bx,by) : Location of the centroid in Grid Space
		#   (bw,bh) : Shape of the bbox in Grid Space
		#   (Cx,Cy) : The cells that exist fromo the top left cornert till the cell where the
		#             centroid was predicted
		#   (Pw,Ph) : The width and height of the anchor box in which the object was detected
		#
		#######
		txy = tf.sigmoid(pred[..., 0:2])       # Shape [?,13,13,5,2]
		tx = txy[...,0]
		ty = txy[...,1]

		twh = tf.exp(pred[...,2:4])            # Shape [?,13,13,5,2]
		tw = twh[...,0]
		th = twh[...,1]

		pobj = tf.sigmoid(pred[...,4])       # Shape [?,13,13,5,1]
		pclass = tf.nn.softmax(pred[...,5:])   # Shape [?,13,13,5,20]

		# Lets get the bx and by
		coord_y = tf.range(self.GRID_H)
		coord_x = tf.range(self.GRID_W)

		Cx,Cy = tf.meshgrid(coord_x,coord_y)
		Cx = tf.cast(Cx,tf.float32)
		Cy = tf.cast(Cy,tf.float32)
		
		##### Note: From here, everything could be in less lines.
		#####       Instead of treat x,y separately, stack them and
		#####       do a single operation for both.

		## In here we are going to condition the Cx,Cy so instead of being of
		## shape [13,13] it will add an extra dimension at the end so it can be
		## broadcasted with tx,ty and summed up happily
		Cx = tf.reshape(Cx,[self.GRID_H,self.GRID_W,1])
		Cy = tf.reshape(Cy,[self.GRID_H,self.GRID_W,1])

		bx = tf.add(tx,Cx)
		by = tf.add(ty,Cy)


		# At this point bx and by are in Grid Space, lets convert it to Image space
		## First lets normalize it, [0,1]
		bx = tf.div(bx,tf.constant(self.GRID_W,dtype=tf.float32))
		by = tf.div(by,tf.constant(self.GRID_H,dtype=tf.float32))

		## Now, lets convert it to Image Space
		bx = tf.multiply(bx,tf.constant(self.IM_W,dtype=tf.float32),name='bx')
		by = tf.multiply(by,tf.constant(self.IM_H,dtype=tf.float32),name='by')

		bb_centroid = tf.concat([tf.expand_dims(by,axis=-1),tf.expand_dims(bx,axis=-1)],axis=-1,name='bb_centroid_yx')


		# Lets get bw,bh now
		Pw = tf.cast(tf.constant(self.ANCHORS[:,1]),tf.float32,name='Pw')
		bw = tf.multiply(tw,Pw)

		Ph = tf.cast(tf.constant(self.ANCHORS[:,0]),tf.float32,name='Ph')
		bh = tf.multiply(th,Ph)

		# At this point bw and bh are in Grid Space, lets convert it to Image space
		## First lets normalize it, [0,1]
		#bw = tf.div(bw,tf.constant(self.GRID_W,dtype=tf.float32))
		#bh = tf.div(bh,tf.constant(self.GRID_H,dtype=tf.float32))

		
		## Now, lets convert it to Image Space
		bw = tf.multiply(bw,tf.constant(self.IM_W,dtype=tf.float32),name='bw')
		bh = tf.multiply(bh,tf.constant(self.IM_H,dtype=tf.float32),name='bh')
		bb_shape = tf.concat([tf.expand_dims(bh,axis=-1),tf.expand_dims(bw,axis=-1)],axis=-1,name='bb_shape_hw')

		# Lets get the class with maximum prob
		self.bb_lbs = tf.argmax(pclass,axis=-1,name='lbs')


		# Lets make a mask to identify which of all the cells actually have 
		# a detected object
		ob_mask = tf.greater(pobj,tf.constant(self.THRESHOLD_OUT_PROB))
		#ob_mask = tf.multiply(tf.cast(ob_mask,tf.float32),pobj)
		self.bb_ob_mask = tf.reshape(ob_mask,[-1,self.GRID_H,self.GRID_W,self.NUM_ANCHORS],name='bb_ob_mask')


		# A final touch, lets make available the prob of each object
		self.pobj = tf.identity(pobj)



		##### Lets just give coordinates
		bbox_min = bb_centroid - (bb_shape//2)
		bbox_max = bb_centroid + (bb_shape//2)

		self.coord = tf.concat([bbox_min[...,0:1], bbox_min[...,1:2], bbox_max[...,0:1], bbox_max[...,1:2]], axis=-1)
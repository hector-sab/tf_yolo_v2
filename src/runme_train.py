import tensorflow as tf
from model import Model
from train import Trainer,Data

if __name__=='__main__':
	ims_paths = '../test_train/0_ims_paths.txt'
	lbs_paths = '../test_train/0_lbs_paths.txt'

	model = Model(init=False)
	train_set = Data(ims_paths,lbs_paths)
	trainer = Trainer(model,train_set,init=True)
	#trainer.optimize(n_iter=1,bs=1)

	"""
	vall = tf.global_variables()

	with open('all_variables_in_train_graph.txt','w') as f:
		for v in vall:
			f.write(v.name+'\n')
	"""
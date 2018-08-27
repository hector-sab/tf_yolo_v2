import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import utils as ut


def plot3D(X,Y,Z,title='',xlabel='',ylabel='',zlabel=''):
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	fig.canvas.set_window_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,
	linewidth=0,antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)

	return(surf)



fdir = '/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/416_416/test_01/'
fname = 'full_results.pk'

if __name__=='__main__':
	np.set_printoptions(linewidth=200,precision=2)
	# Lets count how many objects are in the dataset
	total_true,count = ut.count_total_objects('/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/416_416/00000001/')
	print('Total Number of Objects:',total_true)
	for key in list(count.keys()):
		print('Class {}: {}'.format(key,count[key]))

	# Controls the prob th in the prediction model
	pTH = np.linspace(0,1,21)

	# Controls minimum iou th for the nms
	sTH = np.linspace(0,1,21)

	# Load the results
	results = pickle.load(open(os.path.join(fdir,fname),'rb'))
	rname = 'p-{}_s-{}'

	# Will be charts
	correc_15_Z = np.zeros((pTH.shape[0],sTH.shape[0]))
	gt_no_pred_15_Z = np.zeros((pTH.shape[0],sTH.shape[0]))
	pred_no_gt_15_Z = np.zeros((pTH.shape[0],sTH.shape[0]))

	num_classes_Z = np.zeros((pTH.shape[0],sTH.shape[0]))

	recall = np.zeros((pTH.shape[0],sTH.shape[0]))
	precision = np.zeros((pTH.shape[0],sTH.shape[0]))


	pTHl = ['{:.2f}'.format(x) for x in pTH]
	sTHl = ['{:.2f}'.format(x) for x in sTH]

	for i,p in enumerate(pTHl):
		for j,s in enumerate(sTHl):
			# Create the name of the result to be analysed
			cresult = rname.format(p,s)
			# Loads the result
			out = results[cresult]
			# How many classes detected
			num_classes = len(out)

			num_classes_Z[i,j] = num_classes


			correc_15_Z[i,j] = out[15]['correspondence']
			gt_no_pred_15_Z[i,j] = out[15]['no_correspondence_with_gt'] # Pred qit no gt
			# If new full_results.pk file is generated, use 'no_correspondece_with_pred'
			pred_no_gt_15_Z[i,j] = out[15]['no_correspondece_with_pred']


			tp = out[15]['correspondence']
			fn = out[15]['no_correspondece_with_pred']
			recall[i,j] = tp/(tp+fn+1e-20)
			precision[i,j] = tp/total_true # tp/total_positives

	print('\nX axis: IoU of objects')
	print('Y axis: Prob of objects')
	print('\nNumber of Objects Correctly Predicted:')
	print(correc_15_Z)
	print('\nNumber of Objects no Predicted')
	print(gt_no_pred_15_Z)
	# print('\nNumber of non-Existing Objects Predicted')
	# print(pred_no_gt_15_Z)
	print('\nNumber of Classes Found')
	print(num_classes_Z)
	
	####### Plots
	
	X,Y = np.meshgrid(pTH,sTH)


	# Figure of number of classes detected
	#fig1 = plot3D(X,Y,num_classes_Z,title='Number of Classes Found',
	#	xlabel='Iou',ylabel='Prob',zlabel='# Classes')

	# Figure of number of correct pred for class 15
	# fig2 = plot3D(X,Y,correc_15_Z,title='Number of Objects Correctly Predicted',
	# 	xlabel='Iou',ylabel='Prob',zlabel='# Correct Found')

	# Figure of number of gt with no pred for class 15
	# fig3 = plot3D(X,Y,gt_no_pred_15_Z,title='Number of Objects no Predicted',
	# 	xlabel='Iou',ylabel='Prob',zlabel='# GT with no pred')

	# Figure of number of pred with no gt for class 15
	# fig4 = plot3D(X,Y,pred_no_gt_15_Z,title='Number of non-Existing Objects Predicted',
	# 	xlabel='Iou',ylabel='Prob',zlabel='# PRED with no gt')

	# Figure of recall
	# fig5 = plot3D(X,Y,recall,title='Recall',
	# 	xlabel='Iou',ylabel='Prob',zlabel='tp/(tp+tn)')

	# Figure of precision
	# fig6 = plot3D(X,Y,precision,title='Precision',
	#	xlabel='Iou',ylabel='Prob',zlabel='tp/all_positives')

	# ROC Curve
	# fig7 = plot3D(precision,Y,recall,title='ROC Curve',
	# 	xlabel='Precision',ylabel='Prob',zlabel='Recall')

	i = 10
	tmp_x = precision[i]
	tmp_y = recall[i]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(tmp_y,tmp_x)

	plt.show(block=False)


	# Calculate ROC curve
	# Basics: y axis -> True Positive Rate, which is how many positive
	#  objects were detected / all the true objects
	#         x axis -> False positive Rate, which is how many false
	#  positive objects were detected / all the false objects
	#
	# Recall or Sensitivity
	# y = tp/(tp+tn)
	#
	# Precision
	# x = tp/all_true
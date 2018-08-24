import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


fdir = '/home/hectorsab/data/Documents/CICATA/yolo/CICATA_dataset/416_416/test_01/'
fname = 'full_results.pk'

if __name__=='__main__':
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
			gt_no_pred_15_Z[i,j] = out[15]['no_correspondence_with_gt']
			# If new full_results.pk file is generated, use 'no_correspondece_with_pred'
			pred_no_gt_15_Z[i,j] = out[15]['no_correspondece_with_pred']



	X,Y = np.meshgrid(pTH,sTH)

	# Figure of number of classes detected
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	fig.canvas.set_window_title('Number of Classes Found')
	ax.set_xlabel('Prob')
	ax.set_ylabel('IoU')
	ax.set_zlabel('# Classes')

	surf = ax.plot_surface(X,Y,num_classes_Z,cmap=cm.coolwarm,
		linewidth=0,antialiased=False)


	# Figure of number of correct pred for class 15
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111,projection='3d')
	fig2.canvas.set_window_title('Number of Objects Correctly Predicted')
	ax2.set_xlabel('Prob')
	ax2.set_ylabel('IoU')
	ax2.set_zlabel('# Correct Found')
	fig2.colorbar(surf, shrink=0.5, aspect=5)

	surf = ax2.plot_surface(X,Y,correc_15_Z,cmap=cm.coolwarm,
		linewidth=0,antialiased=False)

	# Figure of number of gt with no pred for class 15
	fig3 = plt.figure()
	ax3 = fig3.add_subplot(111,projection='3d')
	fig3.canvas.set_window_title('Number of Objects no Predicted')
	ax3.set_xlabel('Prob')
	ax3.set_ylabel('IoU')
	ax3.set_zlabel('# GT with no pred')
	fig3.colorbar(surf, shrink=0.5, aspect=5)

	surf = ax3.plot_surface(X,Y,gt_no_pred_15_Z,cmap=cm.coolwarm,
		linewidth=0,antialiased=False)

	# Figure of number of pred with no gt for class 15
	fig4 = plt.figure()
	ax4 = fig4.add_subplot(111,projection='3d')
	fig4.canvas.set_window_title('Number of non-Existing Objects Predicted')
	ax4.set_xlabel('Prob')
	ax4.set_ylabel('IoU')
	ax4.set_zlabel('# PRED with no gt')
	fig4.colorbar(surf, shrink=0.5, aspect=5)

	surf = ax4.plot_surface(X,Y,pred_no_gt_15_Z,cmap=cm.coolwarm,
		linewidth=0,antialiased=False)

	plt.show()
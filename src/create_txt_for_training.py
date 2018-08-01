import os
import glob
from tqdm import tqdm

if __name__=='__main__':
	data_dir ='/data/HectorSanchez/Test/ppl_CICATA/00000001_rs416/'

	ims = []
	lbs = []
	files = os.listdir(data_dir)

	for i in tqdm(range(len(files))):
		file = files[i]
		if file.endswith('.jpg'):
			ims.append(file)
		elif file.endswith('.txt'):
			lbs.append(file)

	# Make sure that both files have their correspondanse
	# in the other
	fims = open('../test_train/ims.txt','w')
	flbs = open('../test_train/lbs.txt','w')
	for i in tqdm(range(len(ims))):
		if ims[i][:-4]+'.txt' in lbs:
			fims.write(data_dir+ims[i]+'\n')
			flbs.write(data_dir+ims[i][:-4]+'.txt\n')
	fims.close()
	flbs.close()
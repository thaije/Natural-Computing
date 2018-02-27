import numpy as np 
from matplotlib import pyplot as plt
from subprocess import run, PIPE
import os
from pathlib import Path
import negsel
import pdb

## REQUIRES PYTHON3
## Takes very long to run!

def chunk(data, length, overlap=False):
	'''
	Function to split data into chunks
	:param (array_like) 	data 
	:param (int) length 	length of chunks
	:param (bool) overlap 	whether chunks should overlap or not
	:return (array_like) chunked data 
	'''
	chunk_data = []
	step = 1 if overlap else length
	for i in range(0,len(data)-length,step):
		chunk = data[i:i+length]
		chunk_data.append(chunk)
	return chunk_data


## If training data is not yet chunked create new file with chunked data
CHUNK_LEN = 10

#SUBFOLDER = 'snd-cert'
SUBFOLDER = 'snd-unm'
TRAIN_PATH = Path('syscalls/' + SUBFOLDER)
DATA = SUBFOLDER + '.train'
CHUNK_PATH = TRAIN_PATH / 'chunks.train'
if not CHUNK_PATH.exists():
	with open(TRAIN_PATH / DATA, 'r') as f:
		syscalls = f.read()
		syscalls = syscalls.replace('\n','')
	
	chunk_data = chunk(syscalls,CHUNK_LEN)

	with open(CHUNK_PATH, 'w') as f:
		f.writelines("%s\n" % c for c in chunk_data)

## Apply to test data
r = 4
TEST_PATH = Path('syscalls/' + SUBFOLDER)
TEST_CASES = 1
for i in range(1,TEST_CASES+1):
	#read data
	case_data = SUBFOLDER + '.' + str(i) + '.test'
	with open(TEST_PATH / case_data, 'r') as f:
		syscalls = np.array(f.readlines())

	#read labels
	case_labels = SUBFOLDER + '.' + str(i) + '.labels'
	with open(TEST_PATH / case_labels, 'r') as f:
		labels = f.readlines()
	labels = np.array(labels).astype(bool)	


	def call_negsel(data):
		'''
		Helper function to call negative selection
		:param data
		:return scores 
		'''
		test_data = []
		lens = []
		#chunk test data and save number of chunks per line
		for line in data:
			chunks = chunk(line[:-1],CHUNK_LEN)
			lens.append(len(chunks))
			test_data.extend(chunks)

		#apply negative selection	
		res = negsel.negsel(CHUNK_PATH, test_data, CHUNK_LEN, r)
		
		#take mean score per line in test_data
		lens = np.array(lens)
		scores = []
		start = 0
		for l in lens[lens>0]:
			scores.append(np.mean(res[start:start+l]))
			start += l

		return scores
	
	## Apply negative selection to positive and negative samples separately
	pos_scores = call_negsel(syscalls[labels])
	neg_scores = call_negsel(syscalls[~labels])

	sens, spec = negsel.cal_roc(neg_scores, pos_scores)
	auc = negsel.calc_auc(sens,spec)

	## Plot
	print('AUC for r = 4 is {}'.format(auc))

	plt.figure()
	plt.plot(1-spec,sens)
	plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1), color='orange', linestyle='--')
	plt.xlim([-.01,1])
	plt.ylim([-.01,1.01])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve Case ' + SUBFOLDER + str(i))


plt.show()

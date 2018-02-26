import numpy as np 
from matplotlib import pyplot as plt
from subprocess import run, PIPE
import os

## REQUIRES PYTHON3

def negsel(test_file, r):
	'''
	Performs negative selection algorithm by calling negsel2.jar

	Parameters:
	--------
	test_file: str
		filename of file containing test data
	r: int
		max len of contingouos substring matches

	Returns:
	--------
	array_like
		array containing log number of pattern matches
	'''
	args = ['java', '-jar', 'negsel2.jar', '-self', 'english.train', '-n', '10', '-r', str(r), '-c', '-l']	
	with open(test_file, 'r') as f:
		lines = f.read()
		p = run(args, stdout=PIPE, input=lines, encoding='ascii')
		res = np.fromstring(p.stdout.replace('\n',''), dtype=float, sep=' ')
	return res


def cal_roc(scores_true, scores_false):
	'''
	Calculate sensitivity and specificity for roc analysis

	Parameters:
	--------
	scores_true: array_like
		negsel scores for correct class
	scores_false: array_like
		negsel scores for anomalous class

	Returns:
	--------
	array_like
		sensitivity
	array_like
		specificity	
	'''
	scores = np.concatenate([scores_true,scores_false])
	uniques = np.unique(scores)
	roc = np.zeros((len(uniques),2))
	for i,u in enumerate(uniques):
		higher = np.where(scores >= u)[0]
		sens = np.sum(higher >= len(scores_true)) / len(scores_false)
		lower = np.where(scores < u)[0]
		spec = np.sum(lower < len(scores_true)) / len(scores_true)
		roc[i] = [sens,spec]
	return roc[:,0], roc[:,1]

def calc_auc(sensitivity, specificity):
	'''
	Calculate area under the curve
	using trapezoidal approximation for the integral
	
	Parameters:
	--------
	sensitivity: array_like
		sensitivity scores
	specificity: array_like
		specificity scores

	Returns:
	--------
	float
		AUC
	'''
	#return negative value since arrays are
	#sorted in descending order
	return -np.trapz(sensitivity,1-specificity)

#perform negsel on english and tagalog test set for english training data
r = 4
res_eng = negsel('english.test', r)
res_tag = negsel('tagalog.test', r)

#Perform ROC analysis for r = 4
sens, spec = cal_roc(res_eng,res_tag)
auc = calc_auc(sens,spec)
print('AUC for r = 4 is {}'.format(auc))

plt.figure()
plt.plot(1-spec,sens)
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1), color='orange', linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')


## Perform ROC analysis for different values of r

#Plot ROC curves
plt.figure()
plt.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1), color='orange', linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves given r')


aucs = []
for r in range(1,10):
	res_eng = negsel('english.test', r)
	res_tag = negsel('tagalog.test', r)

	sens, spec = cal_roc(res_eng,res_tag)
	plt.plot(1-spec,sens, label='r = {}'.format(r))
	aucs.append(calc_auc(sens,spec))

plt.legend()

#Plot AUC against r
plt.figure()
plt.plot(np.arange(1,10),aucs)
plt.xlabel('r')
plt.ylabel('AUC')
plt.title('AUC for differen values of r')

## Perform negative selection for different distractor languages

#best r
r = 3
res_eng = negsel('english.test', r)
print('{:^20}|{:^8}'.format('Language', 'AUC'))
print('--------------------|--------')
for lang in os.listdir('lang/'):
	res_lang = negsel('lang/' + lang, r)
	sens, spec = cal_roc(res_eng,res_lang)
	auc = calc_auc(sens,spec)
	print('{:^20}|{:^8.4}'.format(lang[:-4],auc))


plt.show()

	


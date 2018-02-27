import numpy as np

# Code from Kai for AUC magic

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

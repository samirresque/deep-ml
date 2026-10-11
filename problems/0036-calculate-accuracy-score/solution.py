import numpy as np

def accuracy_score(y_true, y_pred):
	incorrects = np.sum(np.abs(y_true - y_pred))
	print(incorrects)
	accuracy = 1-incorrects/len(y_true)
	return accuracy
	
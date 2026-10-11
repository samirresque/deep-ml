import numpy as np
def precision(y_true, y_pred):
	# precision metric  = tp / (tp + fp)
	# out of all the preds flagged as +ve how many of them were correct
	positive_pred = 0 # tp + fp
	tp = 0
	for i in range(len(y_true)):
		if y_pred[i]==1:
			positive_pred +=1
		if y_pred[i] == 1 and y_true[i]==1:
			tp += 1
	
	if positive_pred == 0:
		return 0
	return tp/positive_pred

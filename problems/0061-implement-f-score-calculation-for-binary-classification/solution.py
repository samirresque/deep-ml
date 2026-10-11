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

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # recall = tp / (tp + fn) -> how many were tp from the labels that should have been predicted positives
    actual_positives  = 0 
	tn = 0
    tp = 0
	for i in range(len(y_true)):
		if y_true[i]==1:
			actual_positives +=1
		if y_pred[i] == 0 and y_true[i]==1:
			tn += 1
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
	if actual_positives == 0:
		return 0
	return tp/actual_positives

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	pres = precision(y_true, y_pred)
	rec = recall(y_true, y_pred)
	denom = (beta**2 * pres) + rec
	if denom == 0:
		return 0.0
	f_beta = (1+beta**2)*(pres*rec)/denom
	return f_beta 




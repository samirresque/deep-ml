import numpy as np

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

    

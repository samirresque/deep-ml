
import numpy as np

def jaccard_index(y_true, y_pred):
	
	# jaccard index = tp / (# of either y_true or y_pred is true OR union)
	tn_count = 0 # true negatives count
	tp_count = 0 # true positives count
	for i in range(len(y_true)):
		if y_true[i] == 0 and y_pred[i] == 0:
			tn_count += 1
		if y_true[i] == 1 and y_pred[i] == 1:
			tp_count += 1
	num_union = len(y_true) - tn_count
	if num_union == 0:
		return num_union
	result = tp_count/num_union
	return round(result, 3)

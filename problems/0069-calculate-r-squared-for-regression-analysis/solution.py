
import numpy as np

def r_squared(y_true, y_pred):
	# r2 : how much/well x explains the variance in y in regression analysis or how well the model fits the data
	y_true_mean = np.mean(y_true)
	SSR = np.sum(np.square(y_true-y_pred))
	SST = np.sum(np.square(y_true-y_true_mean))
	R2 = 1 - SSR/SST
	return R2


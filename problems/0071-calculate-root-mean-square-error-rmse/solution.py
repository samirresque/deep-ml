
import numpy as np

def rmse(y_true, y_pred):
	rmse_res = np.sqrt(np.mean(np.square(y_true-y_pred)))
	return round(rmse_res,3)

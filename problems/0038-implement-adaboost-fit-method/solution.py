import numpy as np
import math

def adaboost_fit(X, y, n_clf):
	n_samples, n_features = np.shape(X)
	w = np.full(n_samples, (1 / n_samples)) # initial weights
	clfs = [] # optimal classifier !
	eps = 1e-10
	for i in range(n_clf):
		optimal_err = float('inf') # initial error
		optimal = {}
		for f in range(n_features):
			for thrs in np.unique(X[:, f]): # try all threshold for features
				polarity = 1
				pred = np.ones(n_samples) # let all be +1 initially
				pred[X[:, f]< thrs] = -1
				pred = pred*polarity
				err = np.sum(w[pred!=y])

				if err > 0.5: # worse than random choice (coin flip for eg)
					err = 1-err
					polarity = -1

				if err < 1e-12:
					err = 0.0
				if err < optimal_err:
					# update optimal values
					optimal_err = err
					optimal = {'polarity': polarity, 'threshold': thrs.item(), 'feature_index': f}

		
		alpha = 0.5*np.log((1-optimal_err)/(optimal_err+ eps))
		optimal['alpha'] = float(alpha)

		pred = np.ones(n_samples) 
		# pred[optimal['polarity']*X[:, optimal['feature_index']]< optimal['polarity']*optimal['threshold']] = -1
		pred[X[:, optimal['feature_index']]< optimal['threshold']] = -1
		w = w*np.exp(-alpha*y*pred)
		w = w/np.sum(w)
		clfs.append(optimal)
		
	return clfs
    
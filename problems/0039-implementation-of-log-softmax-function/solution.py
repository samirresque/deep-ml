import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# avoids numerical instability when softmax is directly applied due to large numbers. 
	'''
		log(softmax) = log(x) - log(sum of exps)
		= log(x) - max - long(sum of exps - max)
	'''
	scores = np.array(scores)
	max_x = np.max(scores)
	stable_x = scores - max_x
	exps = np.exp(stable_x)
	log_sftmax = stable_x - np.log(np.sum(exps))

	return log_sftmax
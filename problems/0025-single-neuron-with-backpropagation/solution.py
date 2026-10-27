import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	updated_bias = initial_bias
	updated_weights = initial_weights
	alpha = learning_rate
	n = len(features)
	mse_values = []
	for _ in range(epochs):
		z = features@updated_weights + updated_bias # 
		sig_z = 1/(1+np.exp(-z))
		mse_values.append(np.mean(float(np.mean((sig_z-labels)**2))))
		# grad = d_sigma = d_mse/d_sigma = 2/n * (sigma - y) 
		grad = (2/n)*(sig_z - labels)  # (n,)

		dz = (grad * sig_z)*(1-sig_z)	# (n,)
		d_weights = np.transpose(features) @ dz		# (n,d).T@(n,)
		d_bias = np.sum(dz) 

		updated_weights -= alpha * d_weights
		updated_bias -= alpha * d_bias

	return updated_weights, updated_bias, mse_values
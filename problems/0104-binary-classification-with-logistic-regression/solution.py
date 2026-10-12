import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	W = weights
	b = bias
	z = X@W + b
	z = np.clip(z, -500, 500)
	probs = 1/(1+np.exp(-z)) # sigmoid = 1/(1+e^-x)

	probs = np.where(probs>=0.5, 1, 0)
	return probs



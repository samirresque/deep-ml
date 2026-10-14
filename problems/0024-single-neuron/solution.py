import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):

	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)

	z = features@weights + bias # forward pass
	probabilities = 1/(1+np.exp(-z))

	mse = np.mean((labels-probabilities)**2)
	return probabilities, mse
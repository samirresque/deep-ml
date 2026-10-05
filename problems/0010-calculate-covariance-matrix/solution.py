import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

	covariance_matrix = np.cov(vectors)
	return covariance_matrix

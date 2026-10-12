import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.array(X)
	# the formula for normal equation theta can be derived using an elementary algebra
	theta = np.linalg.inv(X.T@X)@X.T@y
	return theta
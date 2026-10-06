import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # ensure y is a column vector
    theta = np.zeros((n, 1))  # initialize weights to zeros
    for _ in range(iterations):
        y_hat = X@theta # (m,n)@(n,1) -> (m,1)
        L = 0.5*np.mean((y_hat-y)**2)
        # say dout = dL/dy_hat
        dout = (y_hat-y)/m  #shape (m,1)
        # say dx = dL/dy_hat * dy_hat/dX
        dX = dout@theta.T # (m,1)@(1,n) -> (m,n)
        #say d_theta = dL/dy_hat * dy_hat/d_theta
        d_theta = X.T@dout # (n,m)@(m,1) -> (n,1)
        theta = theta - alpha * d_theta

    return theta.flatten()
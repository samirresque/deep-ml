import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m,n = X.shape # m,n -> examples, features
    if method=='batch':
        batch_size=m
    elif method=='stochastic':
        batch_size=1
    elif method=='mini_batch':
        pass
    else:
        raise ValueError(f'{method} not valid.')
    
    for i in range(n_epochs):
        for batch in range(0, m, batch_size):
            x = X[batch: batch+batch_size] # slice X
            y_ = y[batch: batch+batch_size] # slice y
            y_hat = x@weights # (batch_size,n)@(n,) -> (batch_size,)
            # say dout = dL/dy_hat, where L is MSE
            dout = 2*(y_hat - y_)/batch_size  #shape (batch_size,)
            #say d_theta = dL/dy_hat * dy_hat/d_theta
            d_weights = x.T@dout # (n,batch_size)@(batch_size,) -> (n,)
            weights = weights - learning_rate * d_weights
              
    return weights

    
    

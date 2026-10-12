import numpy as np

def pegasos_kernel_svm(data: np.ndarray, labels: np.ndarray, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """
    n_samples = data.shape[0]
    alphas = np.zeros(n_samples)
    bias = 0.0
    if kernel=='linear':
        K = data @ np.transpose(data) # dot product of every samples, n*n symmetric matrix 
    elif kernel == 'rbf':
        K = np.zeros((n_samples, n_samples)) # (N,N) matrix
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = np.exp((-np.sum((data[i]- data[j])**2))/(2*sigma**2))

    for t in range(1, iterations+1):
        eta = 1/(lambda_val*t) # learning rate for each iteration

        for i, sample in enumerate(data):
            decision = np.sum(alphas*labels*K[i,:]) + bias
            if labels[i]*decision < 1: # margin is violated ?
                alphas[i] = (1-eta*lambda_val)*alphas[i] + eta
                bias = bias + eta*labels[i]
            else: # margin satisfied ?
                alphas[i] = (1-eta*lambda_val)*alphas[i]

    return (alphas, bias)
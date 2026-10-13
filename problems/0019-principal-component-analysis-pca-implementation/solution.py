import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    n_samples, n_features = data.shape
    mean = np.mean(data, axis=0) # mean of columns
    std = np.std(data, axis=0, ddof=1)

    scaled_data = (data - mean)/std
    cov_matrix = np.cov(scaled_data, rowvar=False) # numpy assumes 
    eigen_vals, eigen_vecs =  np.linalg.eigh(cov_matrix)
    idxs = np.argsort(eigen_vals)[::-1] # get the indices of descending eigen_vals
    principal_comps = eigen_vecs[:,idxs]
    principal_comps = principal_comps[:,:k]
    p = np.transpose(principal_comps)
    for i in range(k):
        if p[i][0] < 0:
            # revert the sign of all each axis in 'that' principal component
            p[i] *= -1
    
    principal_comps = np.transpose(p)
    return np.round(principal_comps, 4)

    

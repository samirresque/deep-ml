def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    A = matrix
    A_inv = []
    det_A = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    if det_A == 0:
        # raise ValueError("The given matrix in non-invertible(singular)")
        return None
    A_inv.append([A[1][1]/det_A,-A[0][1]/det_A])
    A_inv.append([-A[1][0]/det_A,A[0][0]/det_A])
    return A_inv
    
import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    
    # ceil mode + partial windows
    # if not stride < kernel_size:
    #     raise ValueError("can't perform overlapping pooling: stride > kernel size")
    N, C, H, W = x.shape

    output_height = int(np.ceil((H-kernel_size)/stride) + 1)
    output_width = int(np.ceil((W-kernel_size)/stride) + 1)

    output_matrix = np.empty((N, C, output_height, output_width), dtype=x.dtype)
    # if not stride <= kernel_size:
    #     raise ValueError("can't perform overlapping pooling: stride > kernel size")

    for i in range(output_height):
		for j in range(output_width):
			r = i*stride
			c = j*stride
            window_height = min(r+kernel_size, H)   # partial max
            window_width = min(c + kernel_size, W)  # '' 
			window = x[:, :, r: window_height, c: window_width]

			output_matrix[:,:,i,j] = np.max(window, axis=(2,3))
    
    return output_matrix

import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	
	x = np.pad(input_matrix, padding) # pad zeros (default) (3x3) -> (5x5)
	out_height = ((input_height + 2*padding - kernel_height)//stride) + 1
	out_width = ((input_width + 2*padding - kernel_width)//stride) + 1

	output_matrix = np.zeros((out_height, out_width))

	# convolution operation
	for i in range(out_height):
		for j in range(out_width):
			r = i*stride
			c = j*stride
			window = x[r: r+kernel_height, c: c+kernel_height]
			output_matrix[i,j] = np.sum(window*kernel)
	
	return output_matrix

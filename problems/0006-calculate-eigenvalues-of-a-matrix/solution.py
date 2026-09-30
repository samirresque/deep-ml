def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	A  = matrix
	det_A = A[0][0]*A[1][1] - A[0][1]*A[1][0]
	trace_A = A[0][0]+A[1][1]
	#lambda_1 and lambda_2 be the eignevalues
	lambda_1 = (trace_A + (trace_A**2-4*det_A)**(1/2))/2
	lambda_2 = (trace_A - (trace_A**2-4*det_A)**(1/2))/2
	eigenvalues = [lambda_1, lambda_2]
	return eigenvalues
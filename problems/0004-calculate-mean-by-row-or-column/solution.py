def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	m = len(matrix) # num of rows
	n = len(matrix[0]) # num of columns
	means = []
	if mode == 'column':
		for j in range(n):
			column_sum = 0
			for i in range(m):
				column_sum += matrix[i][j]
			column_mean = column_sum/m
			means.append(column_mean)
	elif mode == 'row':
		for row in matrix:
			row_sum = sum(row)
			row_mean = row_sum/n
			means.append(row_mean)
	return means
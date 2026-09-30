def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	for i, row in enumerate(matrix):
		scalar_mult = [scalar*i for i in row]
		matrix[i] = scalar_mult
	return matrix
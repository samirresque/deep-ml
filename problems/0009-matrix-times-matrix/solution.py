def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    # c = a@b or dot product of row of a times the dot product of cols of b
    a_shape = (len(a), len(a[0]))
    b_shape = (len(b), len(b[0]))
    c = []
    if a_shape[1] != b_shape[0]:
        return -1
    else:
        for i in range(a_shape[0]): # get each row of a
            c_row = []
            for j in range(b_shape[1]): # get each col of b
                dot_prod = 0
                for k in range(a_shape[1]):
                    dot_prod += a[i][k]*b[k][j]
                c_row.append(dot_prod)
            c.append(c_row)

    return c
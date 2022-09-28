# inverse of matrix algorithm (without numpy)

# Matrix could have any size, but it must be square
# Matrix must be invertible

# Example:
# 1 2 3
# 4 5 6
# 7 8 9

# 1 2 3 1 0 0
# 4 5 6 0 1 0
# 7 8 9 0 0 1
# 1 2 3 1 0 0
# 4 5 6 0 1 0
# 7 8 9 0 0 1


def inverse_of_matrix(matrix):
    # matrix is a list of lists
    # matrix must be invertible
    # matrix must be square
    # matrix must have at least 2 rows and 2 columns

    # add identity matrix to the right of the matrix
    for i in range(len(matrix)):
        matrix[i] += [0] * len(matrix)
        matrix[i][i + len(matrix)] = 1

    # make matrix triangular
    for i in range(len(matrix)):
        # make diagonal element equal to 1
        if matrix[i][i] != 1:
            # if diagonal element is 0, swap rows
            if matrix[i][i] == 0:
                for j in range(i + 1, len(matrix)):
                    if matrix[j][i] != 0:
                        matrix[i], matrix[j] = matrix[j], matrix[i]
                        break
                else:
                    raise ValueError("Matrix is not invertible")

            # divide row by diagonal element
            for j in range(i, len(matrix) * 2):
                matrix[i][j] /= matrix[i][i]

        # make all elements below diagonal equal to 0
        for j in range(i + 1, len(matrix)):
            if matrix[j][i] != 0:
                # subtract row from the row below
                for k in range(i, len(matrix) * 2):
                    matrix[j][k] -= matrix[i][k] * matrix[j][i]

    # make matrix triangular
    for i in range(len(matrix) - 1, -1, -1):
        # make all elements above diagonal equal to 0
        for j in range(i - 1, -1, -1):
            if matrix[j][i] != 0:
                # subtract row from the row above
                for k in range(i, len(matrix) * 2):
                    matrix[j][k] -= matrix[i][k] * matrix[j][i]

    # return inverse of matrix
    return [row[len(matrix):] for row in matrix]


if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(inverse_of_matrix(matrix))



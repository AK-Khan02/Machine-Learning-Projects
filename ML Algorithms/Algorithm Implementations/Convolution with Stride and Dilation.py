def dilate_matrix(matrix, dilation_factor):
    if dilation_factor <= 1:
        return matrix  # No dilation needed.

    original_rows = len(matrix)
    original_cols = len(matrix[0]) if matrix else 0
    new_rows = 1 + (original_rows - 1) * dilation_factor
    new_cols = 1 + (original_cols - 1) * dilation_factor

    # Initialize new matrix with zeros.
    new_matrix = [[0 for _ in range(new_cols)] for _ in range(new_rows)]

    # Copy original matrix elements into new matrix at dilated positions.
    for i in range(original_rows):
        for j in range(original_cols):
            new_matrix[i * dilation_factor][j * dilation_factor] = matrix[i][j]

    return new_matrix

# Example usage
kernel_d = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
dilation_factor = 2
dilated_matrix_d = dilate_matrix(kernel_d, dilation_factor)
print(dilated_matrix_d)

def convolve_matrix(matrix, kernel, stride):
    output_rows = ((len(matrix) - len(kernel)) // stride) + 1
    output_cols = ((len(matrix[0]) - len(kernel[0])) // stride) + 1

    # Initialize the output matrix with zeros.
    output_matrix = [[0 for _ in range(output_cols)] for _ in range(output_rows)]

    # Slide the kernel over the matrix
    for i in range(0, len(matrix) - len(kernel) + 1, stride):
        for j in range(0, len(matrix[0]) - len(kernel[0]) + 1, stride):
            # Element-wise multiplication and sum
            for ki in range(len(kernel)):
                for kj in range(len(kernel[0])):
                    output_matrix[i // stride][j // stride] += matrix[i + ki][j + kj] * kernel[ki][kj]

    return output_matrix

matrix = [[1, 2, 7, 4, 0, 7, 3], [3,3,2,5,2,5,4], [2,1,4,4,5,2,1],
          [6,0,9,8,1,3,0], [4,9,0,7,2,1,8], [2,1,1,2,5,0,9], [9,2,2,1,3,9,2]]

kernel_a = [[1,1,1], [1,0,1], [1,1,1]]
kernel_c = [[1,0,0,0,1], [0,0,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [1,0,0,0,1]]

convoluted_matrix_a = convolve_matrix(matrix, kernel_a, 2)
convoluted_matrix_c = convolve_matrix(matrix, kernel_c, 1)
convoluted_matrix_d = convolve_matrix(matrix, dilated_matrix_d, 2)

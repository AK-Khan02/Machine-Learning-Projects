import numpy as np

# Define the input matrix
input_matrix = np.array([
    [1, 2, 7, 4, 0, 7, 3],
    [3, 3, 2, 5, 2, 5, 4],
    [2, 1, 4, 4, 5, 2, 1],
    [6, 0, 9, 8, 1, 3, 0],
    [4, 9, 0, 7, 2, 1, 8],
    [2, 1, 1, 2, 5, 0, 9],
    [9, 2, 2, 1, 3, 9, 2]
])

# Define the kernel for the convolution
conv_kernel = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Function to perform a convolution with given stride and kernel
def convolution2d(input_matrix, kernel, stride, pad):
    # Add zero padding to the input matrix
    input_padded = np.pad(input_matrix, [(pad, pad), (pad, pad)], mode='constant', constant_values=0)
    
    # Calculate the dimensions of the output matrix
    output_matrix_size = ((input_padded.shape[0] - kernel.shape[0]) // stride) + 1
    output_matrix = np.zeros((output_matrix_size, output_matrix_size))
    
    # Perform convolution
    for y in range(0, output_matrix_size):
        for x in range(0, output_matrix_size):
            output_matrix[y, x] = np.sum(
                kernel * input_padded[y*stride:y*stride+kernel.shape[0], x*stride:x*stride+kernel.shape[1]]
            )
    return output_matrix

# Perform a 3x3 convolution with stride 2
convolution1_output = convolution2d(input_matrix, conv_kernel, stride=2, pad=0)
convolution1_output

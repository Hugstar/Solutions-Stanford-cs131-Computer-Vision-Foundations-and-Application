import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = None
    compressed_size = 0

    # YOUR CODE HERE
    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    u, s, v = np.linalg.svd(image)

    # Select num_values singular values, zero all others
    s[num_values:] = 0

    # Transform s to a diagonal matrix with size (M,N)
    new_s = np.zeros((u.shape[0], s.shape[0]))
    new_s[:s.shape[0], :s.shape[0]] = np.diag(s)

    # Do matrix multiplication U*S*V
    us = np.matmul(u,new_s)
    compressed_image = np.matmul(us,v)

    # We need to keep only first num_values columns from u, diagonal values of s, first num_values columns of v
    compressed_size = np.size(u[:, :num_values]) + np.size(s[:num_values]) + np.size(v[:,:num_values])
    # END YOUR CODE

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    ''' 
    for point n,m in image with kernel size 3 we need: 
    image[n-1][m-1] * kernel[0][0] + image[n-1][m] * kernel[0][1] + image[n+1][m+1] * kernel[0][2] etc'...
    '''
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    # Get the deltas!
    kernel = np.flip(np.flip(kernel,axis=0),axis=1)
    delta_h = int((Hk-1)/2)
    delta_w = int((Wk-1)/2)
    out = np.zeros((Hi, Wi))
    for image_h in range(1, Hi -delta_h):
        for image_w in range(1, Wi - delta_w):
            sum = 0
            for kernel_h in range(-1*delta_h, delta_h+1):
                for kernel_w in range(-1*delta_w, delta_w+1):
                    sum += (image[image_h + kernel_h][image_w+kernel_w] * kernel[kernel_h+delta_h][kernel_w+delta_w])
            out[image_h][image_w] = sum
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:pad_height+H, pad_width:pad_width+W] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE

    # Flip the kernel
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    Hk, Wk = kernel.shape
    delta_h = int((Hk - 1) / 2)
    delta_w = int((Wk - 1) / 2)
    for image_h in range(delta_h, Hi-delta_h):
        for image_w in range(delta_w, Wi-delta_w):
            out[image_h][image_w] = np.sum(kernel*image[image_h-delta_h:image_h+delta_h+1,image_w-delta_w:image_w+delta_w+1])
    ### END YOUR CODE

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE

    # Flip the kernel
    # kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    # Hk, Wk = kernel.shape
    # delta_h = int((Hk - 1) / 2)
    # delta_w = int((Wk - 1) / 2)
    # func = (lambda x,y: np.sum(
    #             kernel * image[x - delta_h:x + delta_h + 1, y - delta_w:y+ delta_w + 1]))
    # TODO: this actually slows the code down! need to check why
    # func = np.vectorize(func)
    # for image_h in range(delta_h, Hi - delta_h):
    #     for image_w in range(delta_w, Wi - delta_w):
    #         out[image_h][image_w] = func(image_h, image_w)

    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

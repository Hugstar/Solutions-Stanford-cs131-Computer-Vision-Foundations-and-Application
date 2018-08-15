import numpy as np
from skimage import feature,data, color, exposure, io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import gaussian
from scipy import signal
from scipy.ndimage import interpolation
import math

def hog_feature(image, pixel_per_cell = 8):
    ''' 
    Compute hog feature for a given image.
    
    Hint: use the hog function provided by skimage
    
    Args:
        image: an image with object that we want to detect
        pixel_per_cell: number of pixels in each cell, an argument for hog descriptor
        
    Returns:
        score: a vector of hog representation
        hogImage: an image representation of hog provided by skimage
    '''
    ### YOUR CODE HERE
    hogFeature, hogImage = feature.hog(image, pixels_per_cell=(pixel_per_cell, pixel_per_cell), visualise=True)
    ### END YOUR CODE
    return (hogFeature, hogImage)

def sliding_window(image, base_score, stepSize, windowSize, pixel_per_cell=8):
    ''' A sliding window that checks each different location in the image, 
        and finds which location has the highest hog score. The hog score is computed
        as the dot product between hog feature of the sliding window and the hog feature
        of the template. It generates a response map where each location of the
        response map is a corresponding score. And you will need to resize the response map
        so that it has the same shape as the image.
    
    Args:
        image: an np array of size (h,w)
        base_score: hog representation of the object you want to find, an array of size (m,)
        stepSize: an int of the step size to move the window
        windowSize: a pair of ints that is the height and width of the window
    Returns:
        max_score: float of the highest hog score 
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        response_map: an np array of size (h,w)
    '''
    # slide a window across the image
    (max_score, maxr, maxc) = (0,0,0)
    winH, winW = windowSize
    print('window size ', windowSize)
    H,W = image.shape
    pad_image = np.lib.pad(image, ((winH//2,winH-winH//2),(winW//2, winW-winW//2)), mode='constant')
    response_map = np.zeros((H//stepSize+1, W//stepSize+1))
    
    ### YOUR CODE HERE
    # Start from 0,0 in pad image/
    for row in np.arange(0, H+1, stepSize):
        # print('row {}'.format(row))
        for col in np.arange(0, W+1, stepSize):
            # print('col {}'.format(col))
            # continue
            img_patch = pad_image[row: row+winH, col: col+winW]
            img_patch_hog = feature.hog(img_patch, pixels_per_cell=(pixel_per_cell, pixel_per_cell))
            assert img_patch_hog.shape == base_score.shape
            score = np.dot(img_patch_hog, base_score)
            response_map[row//stepSize, col//stepSize] = score

            # Update max score if necessary
            if score > max_score:
                max_score = score
                maxc, maxr = col, row
                # print('max - ({},{})'.format(maxr, maxc))

    response_map = resize(response_map, image.shape[0:2])
    assert response_map.shape[0:2] == image.shape[0:2]
    maxr,maxc = maxr-winH//2, maxc-winW//2
    ### END YOUR CODE

    return (max_score, maxr, maxc, response_map)


def pyramid(image, scale=0.9, minSize=(200, 100)):
    '''
    Generate image pyramid using the given image and scale.
    Reducing the size of the image until on of the height or
    width reaches the minimum limit. In the ith iteration, 
    the image is resized to scale^i of the original image.
    
    Args:
        image: np array of (h,w), an image to scale
        scale: float of how much to rescale the image each time
        minSize: pair of ints showing the minimum height and width
        
    Returns:
        images: a list containing pair of 
            (the current scale of the image, resized image)
    '''
    # yield the original image
    images = []
    current_scale = 1.0
    images.append((current_scale, image))
    # keep looping over the pyramid
    ### YOUR CODE HERE
    new_h, new_w = [x * 0.9 for x in images[-1][-1].shape[0:2]]
    while(new_h > minSize[0] and new_w > minSize[1]):
        new_img = rescale(images[-1][-1], scale)
        current_scale *= scale
        images.append((current_scale, new_img))
        print ('added image with shape {}'.format(new_img.shape))
        new_h, new_w = [x * 0.9 for x in images[-1][-1].shape[0:2]]

    ### END YOUR CODE
    return images

def pyramid_score(image,base_score, shape, stepSize=20, scale = 0.9, pixel_per_cell = 8):
    '''
    Calculate the maximum score found in the image pyramid using sliding window.
    
    Args:
        image: np array of (h,w)
        base_score: the hog representation of the object you want to detect
        shape: shape of window you want to use for the sliding_window
        
    Returns:
        max_score: float of the highest hog score 
        maxr: int of row where the max_score is found
        maxc: int of column where the max_score is found
        max_scale: float of scale when the max_score is found
        max_response_map: np array of the response map when max_score is found
    '''
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    max_response_map =np.zeros(image.shape)
    images = pyramid(image, scale)
    ### YOUR CODE HERE
    for curr_image in images:
        score, r, c, rmap = sliding_window(curr_image[-1], base_score, stepSize, shape, pixel_per_cell=pixel_per_cell)
        if score > max_score:
            max_score = score
            maxr = r
            maxc = c
            max_response_map = rmap
            max_scale = curr_image[0]
    ### END YOUR CODE
    max_response_map = resize(max_response_map, image.shape[0:2])
    assert max_response_map.shape[0:2] == image.shape[0:2]
    return max_score, maxr, maxc, max_scale, max_response_map


def compute_displacement(part_centers, face_shape):
    ''' Calculate the mu and sigma for each part. d is the array 
        where each row is the main center (face center) minus the 
        part center. Since in our dataset, the face is the full
        image, face center could be computed by finding the center
        of the image. Vector mu is computed by taking an average from
        the rows of d. And sigma is the standard deviation among 
        among the rows. Note that the heatmap pixels will be shifted 
        by an int, so mu is an int vector.
    
    Args:
        part_centers: np array of shape (n,2) containing centers 
            of one part in each image
        face_shape: (h,w) that indicates the shape of a face
    Returns:
        mu: (1,2) vector
        sigma: (1,2) vector
        
    '''
    # d - each row is the main center (face center) minus the part center
    # Assumption - center of the face is the center of the image
    # mu - average of d rows
    # sigma standard deviation among the rows

    d = np.zeros((part_centers.shape[0],2))
    ### YOUR CODE HERE
    face_center = [x/2 for x in face_shape]
    d = face_center - part_centers
    # print('d', d)
    mu = np.mean(d, axis=0)[np.newaxis,...].astype(np.int64)
    sigma = np.std(d,axis=0)[np.newaxis,...].astype(np.int64)
    assert mu.shape == (1,2), 'mu shape is {}'.format(mu.shape)
    assert sigma.shape == (1, 2), 'sigma shape is {}'.format(sigma.shape)
    ### END YOUR CODE
    return mu, sigma
        
def shift_heatmap(heatmap, mu):
    '''First normalize the heatmap to make sure that all the values 
        are not larger than 1.
        Then shift the heatmap based on the vector mu.

        Args:
            heatmap: np array of (h,w)
            mu: vector array of (1,2)
        Returns:
            new_heatmap: np array of (h,w)
    '''
    new_heatmap = np.zeros_like(heatmap)

    ### YOUR CODE HERE
    # Convert from (1,2) to (2,)
    if np.array(mu).shape == (1,2):
        mu = mu[0]
    # Normalize - divide by max value
    heatmap /= np.max(heatmap)
    mom = lambda s: s if s<0 else None
    non = lambda s: max(0,s)
    # Nice shifting trick that handles also non-positive number, worth taking time to understand :)
    new_heatmap[non(-mu[0]):mom(-mu[0]), non(-mu[1]):mom(-mu[1])] = heatmap[non(-mu[0]):mom(-mu[0]), non(-mu[1]):mom(-mu[1])]
    return new_heatmap
    

def gaussian_heatmap(heatmap_face, heatmaps, sigmas):
    '''
    Apply gaussian filter with the given sigmas to the corresponding heatmap.
    Then add the filtered heatmaps together with the face heatmap.
    Find the index where the maximum value in the heatmap is found. 
    
    Hint: use gaussian function provided by skimage
    
    Args:
        image: np array of (h,w)
        sigma: sigma for the gaussian filter
    Return:
        new_image: an image np array of (h,w) after gaussian convoluted
    '''
    ### YOUR CODE HERE
    max_heatmap_val = - 1
    # print('sigmas', sigmas)
    for curr_heatmap, sigma in zip(heatmaps, sigmas):
        # print('sigma ', sigma)
        for sig in sigma[0]:
            # print('curr_heatmap.shape', curr_heatmap.shape)
            # print('sig', sig)
            curr_filtered_hm = gaussian(curr_heatmap, sigma=sig)
            curr_filtered_hm += heatmap_face
            if np.max(curr_filtered_hm) > max_heatmap_val:
                max_heatmap_val = np.max(curr_filtered_hm)
                r,c = np.where(curr_filtered_hm == curr_filtered_hm.max())
                heatmap = curr_heatmap
    ### END YOUR CODE
    return heatmap, r , c
            
      
def detect_multiple(image, response_map):
    '''
    Extra credit
    '''
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return detected_faces

            

    
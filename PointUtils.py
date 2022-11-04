import numpy as np
import cv2

def AddThickBinaryDots(array, size=(7,7), *args):
    for origin in args:
        kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)) + origin - ((np.asarray(size) - 1) / 2).astype('uint8'))
        array[kernel[:,0], kernel[:,1]] = 1
    
    return array

def PointInNeighborhood(point1, point2, size=(5,5)):
    # Is point1 in the neighbordhood of point2?
    
    # Where point1 and point2 are (y, x) tuples
    # Get the circlular kernel of radius 20 and offset to endpoint coords
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
    neighboring_indices = np.argwhere(kernel) + point2 - ((np.asarray(size) - 1) / 2)
    
    # Check if the point is in neighborhood
    if any(np.equal(neighboring_indices, point1).all(axis=1)):
        return True
    else:
        return False
    
def ContainsMutualPoints(array1, array2, return_array=False):
    """ Determine whether any point in array1 exists within array2

    Args:
        array1 (ndarray): The (n,2) sub-array, consituted of (y,x) points
        array2 (ndarray): The (n,2) principal-array, consituted of (y,x) points

    Returns:
        bool: Does any point in the sub array exist in the principal array
    """
    
    if not isinstance(array1, np.ndarray):
        array1 = np.asarray(array1)
    
    if not isinstance(array2, np.ndarray):
        array2 = np.asarray(array2)
        
    if len(np.shape(array1)) == 1:
        array1 = np.reshape(array1, (1,2))

    # Creates an ndarray<bool> in the shape of array2 where each element indicates whether that element also exists in array1
    bool_overlap_array = np.asarray((array1[:, None] == array2)).all(axis=2).any(axis=0)
    
    if return_array:
        return bool_overlap_array
    else:
        return any(bool_overlap_array)
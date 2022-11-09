import numpy as np
import cv2

def AddThickBinaryDots(array, *args, size=(7,7)):
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
    
def ContainsMutualPoints(array1, array2):
    """ Determine whether any point in array1 exists within array2

    Args:
        array1 (ndarray): The (m,2) sub-array, consituted of (y,x) points
        array2 (ndarray): The (n,2) principal-array, consituted of (y,x) points

    Returns:
        ndarray<bool>: 1D bool (n) array, indicating which rows in array2 also exit within array1
    """
    
    if not isinstance(array1, np.ndarray):
        array1 = np.asarray(array1)
    
    if not isinstance(array2, np.ndarray):
        array2 = np.asarray(array2)
        
    if len(np.shape(array1)) == 1:
        array1 = np.reshape(array1, (1,2))

    # Creates an ndarray<bool> in the shape of array2 where each element indicates whether that element also exists in array1
    bool_overlap_array = np.asarray((array1[:, None] == array2)).all(axis=2).any(axis=0)
    return bool_overlap_array

def Distance(P1, P2):
    """
    This function computes the distance between 2 points defined by
    P1 = (x1,y1) and P2 = (x2,y2) 
    """

    return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5


def OptimizePath(coords, start=None):
    """
    This function finds the nearest point to a point
    coords should be a list in this format coords = [ [x1, y1], [x2, y2] , ...] 

    """
    if start is None:
        start = coords[0]
    pass_by = coords
    path = [start]
    if start in pass_by:
        pass_by.remove(start)
    while pass_by:
        nearest = min(pass_by, key=lambda x: Distance(path[-1], x))
        path.append(nearest)
        pass_by.remove(nearest)
    return path
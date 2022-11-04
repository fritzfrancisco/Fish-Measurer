class TravellingSalesman():
    def __init__(self, array_list):
        self.array_list = array_list
        
    def distance(self, P1, P2):
        """
        This function computes the distance between 2 points defined by
        P1 = (x1,y1) and P2 = (x2,y2) 
        """

        return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5


    def optimized_path(self, coords, start=None):
        """
        This function finds the nearest point to a point
        coords should be a list in this format coords = [ [x1, y1], [x2, y2] , ...] 

        """
        if start is None:
            start = coords[0]
        pass_by = coords
        path = [start]
        pass_by.remove(start)
        while pass_by:
            nearest = min(pass_by, key=lambda x: self.distance(path[-1], x))
            path.append(nearest)
            pass_by.remove(nearest)
        return path
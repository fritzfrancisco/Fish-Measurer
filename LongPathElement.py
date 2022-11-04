class LongPathElement:
    def __init__(self, branch_index, intersection_pts, intersection_pt_indices, end_pts=None, end_pt_indices=None):
        self.branch_index = branch_index
        self.intersection_pts = intersection_pts
        self.intersection_pt_indices = intersection_pt_indices
        self.end_pts = end_pts
        self.end_pt_indices = end_pt_indices
        
        self.head_point = None
        self.tail_point = None
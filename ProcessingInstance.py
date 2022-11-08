import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as u
from skimage import img_as_bool
from skimage.morphology import medial_axis, binary_closing, binary_opening
from scipy.signal import find_peaks
import PointUtils
from LongPathElement import LongPathElement

class ProcessingInstance():
    def __init__(self, process_id, raw_frame, binarized_frame, outputfolder):
        # Direct attributions
        self.process_id = process_id
        self.raw_frame = raw_frame
        self.binarized_frame = binarized_frame
        self.outputfolder = outputfolder

        # Image arrays
        self.skeleton = None
        self.distance_transform = None
        
        self.dimensions = np.shape(self.binarized_frame)
        self.contour = np.zeros(self.dimensions)
        self.long_path_binary = np.zeros(self.dimensions)
        self.processed_frame = np.zeros(self.dimensions)
        self.skeleton_contour = np.zeros(self.dimensions)
        
        # FilFinder data
        self.fil_finder = None
        self.filament = None
        self.fil_length_pixels = 0

        # State information
        self.successful_init = True

        # Longest path variables
        self.longest_path_branch_indices = []
        self.head_point = None
        self.tail_point = None
        self.covered_intersec_pt_indices = []
        self.number_of_branches = None
        self.long_path_pixel_coords = []
        
        # list of tuples (branch_index, intersection_pt_indices)
        self.long_path_elements = []
        
        # Standard length point
        self.standard_length_point = None
        self.slp_near_endpoint = None


        ## -- START -- ##
        # Apply morphological operations (image processing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(binarized_frame, cv2.MORPH_OPEN, kernel)
        # self.contour = cv2.morphologyEx(cv2.erode(opening,kernel,iterations = 1), cv2.MORPH_GRADIENT, kernel)
        contours, hier = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.contour, contours,-1,255)

        # Skeletonize
        bool_image = img_as_bool(binary_opening(opening))
        self.skeleton, self.distance_transform = medial_axis(bool_image, return_distance=True)
        self.skeleton = ((binary_closing(self.skeleton))* 255).astype('uint8')

        # # Visualize the dist transform
        # image = cv2.resize(np.rint(self.distance_transform).astype('uint8'), None, fy=0.3, fx=0.3)
        # cv2.imshow("Dist_map", image)
        # cv2.waitKey(0)

        # FilFinder operations
        self.fil_finder = FilFinder2D(self.skeleton, mask=self.skeleton)
        self.fil_finder.create_mask(verbose=False, use_existing_mask=True)
        self.fil_finder.medskel(verbose=False)

        try:
            self.fil_finder.analyze_skeletons(skel_thresh=1*u.pix)
        except ValueError:
            print("Filfinder error")
            self.successful_init = False

        # Grab the relevant filament
        try:
            if len(self.fil_finder.lengths()) > 1:
                lengths = [q.value for q in self.fil_finder.lengths()]
                index = lengths.index(max(self.fil_finder.lengths()).value)
                self.filament = self.fil_finder.filaments[index]
            else:
                self.filament = self.fil_finder.filaments[0]
        except:
            print("could not grab filament")
            self.successful_init = False

    def ConstructLongestPath(self):
        self.number_of_branches = len(self.filament.branch_properties["length"])
        print("\nSkeleton contains {0} branches".format(self.number_of_branches))

        self.filament.rht_branch_analysis()
        
        # Head point operations
        self.head_point = ProcessingInstance.FindHeadPoint(self.filament, self.distance_transform)
        base_branch_index = ProcessingInstance.GetBranchFromPoint(self.head_point, self.filament)[0]
        print("starting branch is branch {0}".format(base_branch_index))
        self.longest_path_branch_indices.append(base_branch_index)
        
        intersection_points, intersection_indices = ProcessingInstance.GetBranchIntersections(base_branch_index, also_indices=True)
        end_points, end_indices = ProcessingInstance.GetBranchEndPoints(base_branch_index, also_indices=True)
        self.long_path_elements.append(LongPathElement(base_branch_index, intersection_points, intersection_indices, end_points, end_indices))
        
        # Recursively construct the longest path network
        if not self.RecursiveBranching(base_branch_index):
            print("Issue with LP recursion, aborting.")
            return False

        # Can maybe do this at the same time as the recursive branching
        if not self.ConstructOrderedLongPathCoordinates():
            print("Issue with ordering and trimming, aborting.")
            return False
        
        self.long_path_pixel_coords = np.asarray(self.long_path_pixel_coords)
        
        # Ensure that the long path operation was successful and meaningful by checking if
        # it's within some tolerance of the FilFinder value
        if self.fil_length_pixels < 0.7 * self.filament.length(u.pix).value:
            print("Length {0:.2f} failed to be > (0.7 * Filfinder's long_path length) = {1:.2f}".format(self.fil_length_pixels, 0.7 * self.filament.length(u.pix).value))
            return False
        else:
            print("Length {0:.2f} > (0.7 * Filfinder's long_path length) = {1:.2f}".format(self.fil_length_pixels, 0.7 * self.filament.length(u.pix).value))

        # Long path is assembled, start peripheral operations
        print("Finding SLP...")
        if not self.FindStandardLengthPoint():
            print("Issue with SLP, aborting.")
            return False

        print("Adding contour distances...")
        if not self.AddContourDistances():
            return False
        
        # Draw the picture
        self.long_path_binary[self.long_path_pixel_coords[:,0], self.long_path_pixel_coords[:,1]] = 1
        self.long_path_binary = PointUtils.AddThickBinaryDots(self.long_path_binary, self.head_point, self.tail_point)
        if not self.slp_near_endpoint:
            self.long_path_binary = PointUtils.AddThickBinaryDots(self.long_path_binary, self.standard_length_point, size=(5,5))
        
        # The tail of each element is the head of the following element, the last element's tail will be the overall tail point/SLP, which are already covered above
        connectors_list = [element.head_point for element in self.long_path_elements]
        self.long_path_binary = PointUtils.AddThickBinaryDots(self.long_path_binary, *connectors_list, size=(5,5))
        
        # Final touches
        self.skeleton_contour = np.rint((self.long_path_binary * 255 + self.contour)/2).astype('uint8')
        self.processed_frame = cv2.addWeighted(self.skeleton_contour,0.65,cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY).astype('uint8'),0.35,0)
        
        # image = self.processed_frame
        # temp = cv2.resize(image, None, fy=0.5, fx=0.5)
        # cv2.imshow("processed", temp)
        # cv2.waitKey(0)
        
        return True

    def RecursiveBranching(self, base_branch_index):
        if len(self.covered_intersec_pt_indices) == len(self.filament.intersec_pts):
            # There are no more intersection points to evaluate, we've looked at them all
            print("We're done-so's, all intersection points have been assessed")
            if self.long_path_elements[-1].end_pts:
                self.tail_point = [end_point for end_point in self.long_path_elements[-1].end_pts if end_point not in [self.head_point]][0]
                return True
            else:
                print("Did not find the endpoint on the terminal branch")
                return False
        else:
            # Get the raw intersection point(s)
            intersection_points, intersection_indices = ProcessingInstance.GetBranchIntersections(base_branch_index, also_indices=True)
            
            # Remove duplicates and points that have already been covered
            intersec_pts_on_basebranch_indices = []
            
            [intersec_pts_on_basebranch_indices.append(index) for index in intersection_indices if index not in self.covered_intersec_pt_indices and index not in intersec_pts_on_basebranch_indices]
            self.covered_intersec_pt_indices.extend(intersec_pts_on_basebranch_indices)
            print("Indices not yet covered: {0}".format(intersec_pts_on_basebranch_indices))
            
            if not intersec_pts_on_basebranch_indices:
                # The end of the branch must be an endpoint, we're done
                print("We're done-so's, no intersection points found on the base branch")
                if self.long_path_elements[-1].end_pts:
                    self.tail_point = [end_point for end_point in self.long_path_elements[-1].end_pts if end_point not in [self.head_point]][0]
                    return True
                else:
                    print("Did not find the endpoint on the terminal branch")
                    return False
            else:
                # Find which other branches also share this intersection point(s)
                connected_branch_indices = []
                for index in intersec_pts_on_basebranch_indices:
                    contending_intersec_pts = self.filament.intersec_pts[index] if isinstance(self.filament.intersec_pts[index], list) else [self.filament.intersec_pts[index]]
                    for point in contending_intersec_pts:
                        possible_branches = ProcessingInstance.GetBranchFromPoint(point, self.filament, (5,5))
                        if possible_branches:
                            # Add it if it hasn't already been considered, and if it's not already on the long path
                            [connected_branch_indices.append(branch) for branch in possible_branches if branch not in connected_branch_indices and branch not in self.longest_path_branch_indices]
                            
                print("Possible next branches are: {0}".format(connected_branch_indices))

                # Compare the orientations of all contending branches to the base branch to select the next leg
                base_branch_orientation = self.filament.orientation_branches[base_branch_index]

                best_aligned_index = None
                best_aligned_distance = None
                for branch_index in connected_branch_indices:
                    contending_branch_orientation = self.filament.orientation_branches[branch_index]
                    contending_distance = abs(base_branch_orientation - contending_branch_orientation)

                    if best_aligned_index is None:
                        best_aligned_index = branch_index
                        best_aligned_distance = contending_distance
                    else:
                        if contending_distance < best_aligned_distance:
                            print("Branch {0}: {1} is better aligned than branch {2}: {3}".format(branch_index, contending_distance, best_aligned_index, best_aligned_distance))
                            best_aligned_index = branch_index
                            best_aligned_distance = contending_distance

                # Add this branch to the longest path and run through again with an updated base branch
                if best_aligned_index is not None:
                    print("Branch {0} is most aligned, adding".format(best_aligned_index))
                    self.longest_path_branch_indices.append(best_aligned_index)
                    
                    intersection_points, intersection_indices = ProcessingInstance.GetBranchIntersections(best_aligned_index, also_indices=True)
                    end_points, end_indices = ProcessingInstance.GetBranchEndPoints(best_aligned_index, also_indices=True)
                    self.long_path_elements.append(LongPathElement(best_aligned_index, intersection_points, intersection_indices, end_points, end_indices))
                    
                    if self.RecursiveBranching(best_aligned_index):
                        return True
                    else:
                        return False
                else:
                    print("Intersection point exists, but no branches are connected...? Long path failure, aborting.")
                    return False

    def FindStandardLengthPoint(self):
        # Assemble the distances into an ordered 1D array and get minima. long_path_pixel_coords is already ordered
        longpath_distance_array = self.distance_transform[self.long_path_pixel_coords[:,0], self.long_path_pixel_coords[:,1]]
        average_distance = np.average(longpath_distance_array)
        local_minima_indices, _ = find_peaks(-longpath_distance_array, height=-average_distance, prominence=1)
        
        print("Possible SLPs at indices: {0}".format(local_minima_indices))
        print("Corresponding values: {0}".format(self.long_path_pixel_coords[local_minima_indices]))
            
        if local_minima_indices.size > 0:
            # The relevant index, if it exists, will be likely around 60-90% of the way along head --> tail of the fish
            # Within this range, choose the point with the smallest value in the distance transform
            long_path_length = len(self.long_path_pixel_coords)
            lb = round(0.6 * long_path_length)
            ub = round(0.9 * long_path_length)
            print("Feasible bounds: {0}, {1}".format(lb, ub))
            
            feasible_indices = np.asarray(np.where((local_minima_indices >= lb) & (local_minima_indices <= ub)))
            if feasible_indices.size > 0:
                print("Feasible SLPs at indices: {0}".format(local_minima_indices[feasible_indices]))
                mindist_feasible_index = np.argmin(longpath_distance_array[feasible_indices])
                global_index = (local_minima_indices[feasible_indices])[0][mindist_feasible_index]

                self.standard_length_point = self.long_path_pixel_coords[global_index]
                print("Standard length point found at index {2} of {3}: {0}; Tailpoint: {1}".format(self.standard_length_point, self.tail_point, global_index, len(self.long_path_pixel_coords)))
                
                # Check whether the SLP meaningfully exists, or whether it's basically the tail endpoint
                self.slp_near_endpoint = False
                if PointUtils.PointInNeighborhood(self.standard_length_point, self.tail_point, size=(41,41)):
                    self.slp_near_endpoint = True
                    print("SLP =~ Tailpoint")
                else:
                    if not self.TrimSLPLength():
                        return False
                    else:
                        return True
            else:
                print("All feasible SLP potentials weeded out")
                self.standard_length_point = None
                self.slp_near_endpoint = True
        else:
            print("No SLP potentials found")
            self.standard_length_point = None
            self.slp_near_endpoint = True
            
    def TrimSLPLength(self):
        # Which branch of the longest path contains the SLP?
        slp_index = ProcessingInstance.GetBranchFromPoint(self.standard_length_point, self.filament)
        if slp_index:
            long_path_branch_indices = [element.branch_index for element in self.long_path_elements]
            
            slp_branch_index = long_path_branch_indices.index(slp_index)
            print("Long path branch indices: {0}; SLP on branch {1}".format(long_path_branch_indices, slp_index))
            
            # Remove branches in reverse order until that branch
            for long_path_elements_index in range(len(long_path_branch_indices)-1, slp_branch_index, -1):
                self.RemoveBranch(long_path_elements_index)
                
                element = self.long_path_elements[long_path_elements_index]
                print("Removed branch {0} with length {1:.2f}".format(element.branch_index, element.total_adjusted_length))
            
            # Now remove points in the relevant branch up until the SLP
            print("Done removing full branches, current branches on the long path: {0}".format([element.branch_index for element in self.long_path_elements]))
            element = self.long_path_elements[slp_branch_index]
            
            element_branch_points = list(map(tuple, element.ordered_branch_points_adjusted))
            slp_index_on_branch = element_branch_points.index(self.standard_length_point)
            points_to_remove = np.asarray(element_branch_points[slp_index_on_branch+1:])
            length_to_remove = len(points_to_remove) * element.unit_length
            print("Removing {0} points, representing {1:.2f} pixels".format(len(points_to_remove), length_to_remove))
            
            # Reflect these changes in the element itself
            bool_array = PointUtils.ContainsMutualPoints(points_to_remove, element.ordered_branch_points_adjusted)
            element.ordered_branch_points_adjusted = element.ordered_branch_points_adjusted[np.where(np.invert(bool_array))]
            element.total_adjusted_length -= length_to_remove
            element.tail_point = map(tuple, element.ordered_branch_points_adjusted[-1,:])
            
            # Apply these changes to the overall ProcessingInstance
            bool_array = PointUtils.ContainsMutualPoints(points_to_remove, self.long_path_pixel_coords)
            self.long_path_pixel_coords = self.long_path_pixel_coords[np.where(np.invert(bool_array))]
            self.fil_length_pixels -= length_to_remove
            
            return True
        else:
            print("Could not identify SLP branch")
            return False

    def AddContourDistances(self):
        # Get the points that are "pulled back" from the origin point, from which the added distance lines will stem
        # These lines will run through the pullback point, through the origin point, and intersect with the contour of the blob
        pullback_pts = []

        # For the head point, use a circle of radius 20 pixels to find a point further back on the skeleton
        head_pullback_pt = ProcessingInstance.CircleMaskIntersection(self.head_point, self.long_path_pixel_coords, self.dimensions)
        if head_pullback_pt is None:
            print("Head pullback does not intersect with the long path")
            return False
        else:
            pullback_pts.append(head_pullback_pt)
            print("Head pullback point set at {0}".format(head_pullback_pt))

        # For the tail point, use the standard length point if it's not within a 20 pixel radius of the end point
        # otherwise, also use a circle mask
        if (not self.slp_near_endpoint):
            pullback_pts.append(self.standard_length_point)
            print("Using standard length point for tail pullback, at {0}".format(self.standard_length_point))
        else:
            tail_pullback_pt = ProcessingInstance.CircleMaskIntersection(self.tail_point, self.long_path_pixel_coords, self.dimensions)
            if tail_pullback_pt is None:
                print("Tail pullback does not intersect with the long path")
                return False
            else:
                pullback_pts.append(tail_pullback_pt)
                print("Using pullback for tail. Tail pullback point set at {0}".format(tail_pullback_pt))
            
        # Run the assessment for each pullback/endpoint pair
        relevant_line_mask = np.zeros(self.dimensions)
        for i in range(2):
            # First create the line mask through the pullback/end points
            line_mask = np.zeros(self.dimensions)
            pullback_pt = pullback_pts[i]
            end_pt = self.head_point if i == 0 else self.tail_point

            slope = (end_pt[0] - pullback_pt[0]) / (end_pt[1] -  pullback_pt[1])
            if np.isinf(slope):
                line_mask[:, pullback_pt[1]] = 1
            else:
                b = end_pt[0] - slope * end_pt[1]

                for x in range(self.dimensions[1]):
                    y_prev = round(slope * (x - 1) + b)
                    y = round(slope * x + b)
                    y_next = round(slope * (x + 1) + b)

                    if y < self.dimensions[0] - 1 and y >= 0:
                        line_mask[y][x] = 1

                    # Fill in as many pixels as needed to keep a continuous line,
                    # keeping to a 50% tributary width (hence dividing by 2)
                    # otherwise we have a dotted line
                    if y_prev != y_next:
                        y_start = y_prev + round((y - y_prev) / 2)
                        y_end = y + round((y_next - y) / 2)

                        y_step = 1 if y_start <= y_end else -1
                        for y in range(y_start, y_end, y_step):
                            if y < self.dimensions[0] - 1 and y >= 0:
                                line_mask[y][x] = 1

            # Find where the thresholded fish boundary and the line mask intersect
            # There will be multiple points since the contour is not one-pixel thick, and an infinite line
            # originating in an enclosed shape will necessarily cross it at least twice
            combined_array = np.add(line_mask, self.contour / 255)
            boundary_intersec_pts = list(zip(*np.where(combined_array > 1.5)))

            # Get minimum distance from end point to contour to select the correct intersection
            reference_end_pt = np.array([end_pt[0], end_pt[1]])
            minimum_distance = None
            closest_boundary_point = None
            for y, x in boundary_intersec_pts:
                coord = np.array([y, x])
                dist = np.linalg.norm(coord - reference_end_pt) # always >= 0
                if minimum_distance is None:
                    minimum_distance = dist
                    closest_boundary_point = coord
                elif minimum_distance > dist:
                    minimum_distance = dist
                    closest_boundary_point = coord

            # Draw the line
            distance = 0
            if i == 0:
                print("Closest boundary point to HEAD endpoint {0}, is {1}".format(end_pt, closest_boundary_point))
                
                # Get the slicing direction right, because numpy's an idiot
                step_y = 1 if end_pt[0]<=closest_boundary_point[0] else -1
                step_x = 1 if end_pt[1]<=closest_boundary_point[1] else -1
                if end_pt[1] == closest_boundary_point[1]:
                    relevant_line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y, end_pt[1]] = \
                    line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y, end_pt[1]]
                elif end_pt[0] == closest_boundary_point[0]:
                    relevant_line_mask[end_pt[0], end_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                    line_mask[end_pt[0], end_pt[1]:closest_boundary_point[1]+step_x:step_x]
                else:
                    relevant_line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                        end_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                    line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                            end_pt[1]:closest_boundary_point[1]+step_x:step_x]
                
                distance = np.linalg.norm(np.asarray(end_pt) - np.asarray(closest_boundary_point))
                print("Added head line to path, measuring {0:.2f} pixels".format(distance))
                
                # image = cv2.resize(current_array, None, fy=0.5, fx=0.5)
                # cv2.imshow("head added", image)
                # cv2.waitKey(0)
                    
            else:
                # Assessing the tail
                # If using the SLP, draw from the SLP to the closest boundary point. Otherwise, draw from tail endpoint
                if not self.slp_near_endpoint:
                    print("Closest boundary point to TAIL SLP {0}, is {1}".format(pullback_pt, closest_boundary_point))
                    step_y = 1 if pullback_pt[0]<=closest_boundary_point[0] else -1
                    step_x = 1 if pullback_pt[1]<=closest_boundary_point[1] else -1
                    if pullback_pt[1] == closest_boundary_point[1]:
                        relevant_line_mask[pullback_pt[0]:closest_boundary_point[0]+step_y:step_y, pullback_pt[1]] = \
                        line_mask[pullback_pt[0]:closest_boundary_point[0]+step_y:step_y, pullback_pt[1]]
                    elif pullback_pt[0] == closest_boundary_point[0]:
                        relevant_line_mask[pullback_pt[0], pullback_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                        line_mask[pullback_pt[0], pullback_pt[1]:closest_boundary_point[1]+step_x:step_x]
                    else:
                        relevant_line_mask[pullback_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                            pullback_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                        line_mask[pullback_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                pullback_pt[1]:closest_boundary_point[1]+step_x:step_x]
                    
                    distance = np.linalg.norm(np.asarray(pullback_pt) - np.asarray(closest_boundary_point))
                else:
                    print("Closest boundary point to TAIL endpoint {0}, is {1}".format(end_pt, closest_boundary_point))
                    step_y = 1 if end_pt[0]<=closest_boundary_point[0] else -1
                    step_x = 1 if end_pt[1]<=closest_boundary_point[1] else -1
                    if end_pt[1] == closest_boundary_point[1]:
                        relevant_line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y, end_pt[1]] = \
                        line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y, end_pt[1]]
                    elif end_pt[0] == closest_boundary_point[0]:
                        relevant_line_mask[end_pt[0], end_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                        line_mask[end_pt[0], end_pt[1]:closest_boundary_point[1]+step_x:step_x]
                    else:
                        relevant_line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                            end_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                        line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                end_pt[1]:closest_boundary_point[1]+step_x:step_x]
                    
                    distance = np.linalg.norm(np.asarray(end_pt) - np.asarray(closest_boundary_point))
                
                print("Added tail line to path, measuring {0:.2f} pixels".format(distance))

            self.fil_length_pixels += distance
        
        # Coerce the binary array of added line segments to bool, and then add the true coordinates to the path
        bool_coerced_additions = np.array(relevant_line_mask, dtype=bool)
        self.long_path_pixel_coords = np.vstack((self.long_path_pixel_coords, np.where(bool_coerced_additions)))
        
        return True
        
    def CircleMaskIntersection(point, array, dimensions):
        """Creates a circle mask of supplied radius centered on the supplied point, and finds where this circle intersects with the supplied array

        Args:
            point (tuple): The (y,x) coordinate that will define the center of the circle to be drawn
            array (np.ndarray): The reference array, which we'll contrast against the drawn circle for intersections

        Returns:
            ndarray<int, int>: An ndarray of size (n,2), where n is the number of intersections found with the circle mask, and each row corresponds to a (y,x) coordinate
        """
        
        circle_mask = np.zeros(dimensions)
        circle_radius = 20
        for x in range(point[1] - circle_radius, point[1] + circle_radius + 1):
            for y in range(point[0] - circle_radius, point[0] + circle_radius + 1):
                if x <= dimensions[1] - 1 and x >= 0:
                    if y <= dimensions[0] - 1 and y >= 0:
                        ref_val = ((x-point[1])**2 + (y-point[0])**2)**(0.5)
                        if ref_val >= circle_radius - 0.7 and ref_val <= circle_radius + 0.7:
                            circle_mask[y][x] = 1
        
        longpath_bool_array = PointUtils.ContainsMutualPoints(np.argwhere(circle_mask), array)
        if not any(longpath_bool_array):
            return None
        else:
            head_pullback_pt = array[np.argwhere(longpath_bool_array)[0], :]
            return head_pullback_pt

    def GetBranchFromPoint(point, filament, size=None):
        """Get the local branch index of the supplied filament that a given point pertains to

        Args:
            point (tuple): a (y,x) tuple
            filament (Filfinder filament): The filament in which the branches to be searched thorugh exist
            size (tuple, optional): An (n,n) tuple that layers a search kernel of the indicated size upon the supplied point. Defaults to None.

        Returns:
            list: A list of branch indices local to the supplied filament that the suplied point belongs to
        """
        # Supply a point (x, y), and get the branch that point belongs to if it exists
        branches = []
        number_of_branches = len(filament.branch_properties["length"])
        if size is None:
            for i in range(number_of_branches):
                if any(PointUtils.ContainsMutualPoints(np.asarray(point), filament.branch_pts(True)[i])):
                    return [i]
        else:
            kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)) + point - ((np.asarray(size) - 1) / 2).astype('uint8'))
            for i in range(number_of_branches):
                if any(PointUtils.ContainsMutualPoints(kernel, filament.branch_pts(True)[i])):
                    branches.append(i)
        
        return branches
            
    def GetBranchIntersections(branch_index, filament, also_indices=False):
        """ Find the coordinates of the supplied filament's intersection points upon the branch at the supplied branch index, if any

        Args:
            branch_index (int): The local index of the branch of the supplied filament object, upon which the search for intersections will take place
            filament (Filfinder filament): The filament, containing branches and intersection points
            also_indices (bool, optional): Whether to return an additional variable with the intersection indices (local to the supplied filament) that were identified as existing on the branch at the supplied index in addition to the intersection coordinates themselves. Defaults to False.

        Returns:
            list<tuple>: A list of (y,x) tuples representing the coordinates of intersections (local to the supplied filament) found on the branch at the supplied index
            OR (list<tuple>, list<int>): A tuple of lists, where the first item is the list defined above, and the second is a 1D list of integers corresponding to the indices local to the filament intersections that were found on the branch at the supplied index
        """
        # Get the intersection point(s) on the branch
        intersec_pts_on_branch = []
        intersec_pts_on_branch_indices = []

        for i in range(len(filament.intersec_pts)):
            intersec_pts = filament.intersec_pts[i] if isinstance(filament.intersec_pts[i], list) else [filament.intersec_pts[i]]
            for point in intersec_pts:
                kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) + point - (2,2))
                branch_mutual_points = any(PointUtils.ContainsMutualPoints(kernel, filament.branch_pts(True)[branch_index]))

                if branch_mutual_points:
                    intersec_pts_on_branch.append(point)
                    intersec_pts_on_branch_indices.append(i)
        
        print("Branch at index: {0} contains the following intersection points: {1}".format(branch_index, intersec_pts_on_branch))
        if not also_indices:
            return intersec_pts_on_branch
        else:
            return intersec_pts_on_branch, intersec_pts_on_branch_indices

    def GetBranchEndPoints(branch_index, filament, also_indices=False):
        """ Find the coordinates of the supplied filament's end points upon the branch at the supplied branch index, if any

        Args:
            branch_index (int): The local index of the branch of the supplied filament object, upon which the search for the end points will take place
            filament (Filfinder filament): The Filfinder filament object, containing branches and end points
            also_indices (bool, optional): Whether to return an additional variable with the end point indices (local to the supplied filament) that were identified as existing on the branch at the supplied index in addition to the end coordinates themselves. Defaults to False.

        Returns:
            list<tuple>: A list of (y,x) tuples representing the coordinates of the end points (local to the supplied filament) found on the branch at the supplied index
            OR (list<tuple>, list<int>): A tuple of lists, where the first item is the list defined above, and the second is a 1D list of integers corresponding to the indices local to the filament end points that were found on the branch at the supplied index
        """
        # Get the intersection point(s) on the branch
        end_pts_on_branch = []
        end_pts_on_branch_indices = []
        
        for i in range(len(filament.end_pts)):
            end_pt = filament.end_pts[i]
            if any(PointUtils.ContainsMutualPoints(np.asarray(end_pt), filament.branch_pts(True)[branch_index])):
                end_pts_on_branch.append(end_pt)
                end_pts_on_branch_indices.append(i)
                        
        print("Branch at index: {0} contains the following end points: {1}".format(branch_index, end_pts_on_branch))
        if not also_indices:
            return end_pts_on_branch
        else:
            return end_pts_on_branch, end_pts_on_branch_indices

    def FindHeadPoint(filament, distance_transform):
        """ Find the head point of the skeleton, defined as the end point with maximal distance to the fish boundary
        
        Args:
            filament (Filfinder filament): The filament in which the branches to be searched thorugh exist
            distance_transform (np.ndarray): An array of size self.dimensions, at every cell of which is the distance to the fish boundary

        Returns:
            tuple: The (y,x) coordinate of the head point set in the dimensional space of self.dimensions
        """
        # Find the starting point: the skeleton end point in the head
        contending_pt = None
        head_point = None
        for i in range(len(filament.end_pts)):

            # The entry may be a list of closely located points, just take the first one if so
            if isinstance(filament.end_pts[i], list):
                contending_pt = filament.end_pts[i][0]
            else:
                contending_pt = filament.end_pts[i]

            if head_point is None:
                head_point = contending_pt
            else:
                print("head_point dist: {0}; contending_pt dist: {1}".format(distance_transform[head_point], distance_transform[contending_pt]))
                if distance_transform[contending_pt] > distance_transform[head_point]:
                    head_point = contending_pt

        print("Head point: {0}".format(head_point))
        return head_point
        
    def ConstructOrderedLongPathCoordinates(self):
        print("Constructing pixel list")
        
        if len(self.long_path_elements) == 1:
            print("Simple construction")
            element = self.long_path_elements[0]
            element.head_point = self.head_point
            element.tail_point = self.tail_point
            
            if not self.AddBranch(element):
                return False
        else:
            print("Advanced construction")
            for i in range(len(self.long_path_elements)):
                element = self.long_path_elements[i]
                
                if i==0:
                    element.head_point = self.head_point
                else:
                    base_element = self.long_path_elements[i-1]
                    print("Considering base branch {0}".format(base_element.branch_index))
                    
                    # Get the shared intersection point
                    shared_intersection = None
                    for base_intersec_point in base_element.intersection_pts:
                        for contender_intersec_point in self.long_path_elements[i].intersection_pts:
                            if (PointUtils.PointInNeighborhood(contender_intersec_point, base_intersec_point)):
                                print("Found shared intersection point at {0}".format(base_intersec_point))
                                shared_intersection = base_intersec_point
                                break
                            
                        if shared_intersection:
                            break
                    
                    if not shared_intersection:
                        return False
                    
                    # Perform base branch operations since both its head and tail are now known
                    # Start by orienting the base branch from head to tail
                    base_element.tail_point = shared_intersection
                    element.head_point = shared_intersection
                    
                    if not self.AddBranch(base_element):
                        return False

                    if i == len(self.long_path_elements)-1:
                        # We're in the final branch and need to handle this branch also
                        print("Adding final branch")
                        element.tail_point = element.end_pts[-1]
                        if not element.tail_point:
                            return False
                        
                        if not self.AddBranch(element):
                            return False
                    
        return True
                    
    def AddBranch(self, element):
        base_branch_points = self.filament.branch_pts(True)[element.branch_index]
        branch_length = self.filament.branch_properties["length"][element.branch_index].value
        if not element.ProcessElement(base_branch_points, branch_length):
            return False
        else:
            # Record the length and save the coords
            self.long_path_pixel_coords.extend(list(map(tuple, element.ordered_branch_points_adjusted)))
            self.fil_length_pixels += element.total_adjusted_length
            
            return True
    
    def RemoveBranch(self, index):
        element = self.long_path_elements[index]
        
        # Remove associated points and length
        bool_array = PointUtils.ContainsMutualPoints(element.ordered_branch_points_adjusted, self.long_path_pixel_coords)
        self.long_path_pixel_coords = self.long_path_pixel_coords[np.where(np.invert(bool_array))]
        self.fil_length_pixels -= element.total_adjusted_length
        
        # Finally, remove this from the long path element list
        del self.long_path_elements[index]
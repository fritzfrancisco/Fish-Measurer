import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as u
from skimage import img_as_bool
from skimage.morphology import medial_axis, binary_closing, binary_opening
from scipy.signal import find_peaks
import os
import PointUtils
from LongPathElement import LongPathElement

class Measurement():
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
        self.FindHeadPoint()
        base_branch_index = self.GetBranchFromPoint(self.head_point)[0]
        print("starting branch is branch {0}".format(base_branch_index))
        self.longest_path_branch_indices.append(base_branch_index)
        
        intersection_points, intersection_indices = self.GetBranchIntersections(base_branch_index, also_indices=True)
        end_points, end_indices = self.GetBranchEndPoints(base_branch_index, also_indices=True)
        self.long_path_elements.append(LongPathElement(base_branch_index, intersection_points, intersection_indices, end_points, end_indices))
        
        if not self.RecursiveBranching(base_branch_index):
            return False

        # Can maybe do this at the same time as the recursive branching
        if not self.ConstructOrderedLongPathCoordinates():
            return False
        
        self.long_path_pixel_coords = np.asarray(self.long_path_pixel_coords)
        self.long_path_binary[self.long_path_pixel_coords[:,0], self.long_path_pixel_coords[:,1]] = 1
        
        # Ensure that the long path operation was successful and meaningful by checking if
        # it's within some tolerance of the FilFinder value
        if self.fil_length_pixels < 0.7 * self.filament.length(u.pix).value:
            print("Length {0:.2f} failed to be > (0.7 * Filfinder's long_path length) = {1:.2f}".format(self.fil_length_pixels, 0.7 * self.filament.length(u.pix).value))
            return False
        else:
            print("Length {0:.2f} > (0.7 * Filfinder's long_path length) = {1:.2f}".format(self.fil_length_pixels, 0.7 * self.filament.length(u.pix).value))

        # Long path is assembled, start peripheral operations
        print("Finding SLP...")
        self.FindStandardLengthPoint()

        print("Adding contour distances...")
        if not self.AddBoundaryDistances():
            return False
        
        # Add base skeleton ending point information
        self.long_path_binary = PointUtils.AddThickBinaryDots(self.long_path_binary, self.head_point, self.tail_point)
        
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
            intersection_points, intersection_indices = self.GetBranchIntersections(base_branch_index, also_indices=True)
            
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
                        possible_branches = self.GetBranchFromPoint(point, (5,5))
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
                    
                    intersection_points, intersection_indices = self.GetBranchIntersections(best_aligned_index, also_indices=True)
                    end_points, end_indices = self.GetBranchEndPoints(best_aligned_index, also_indices=True)
                    self.long_path_elements.append(LongPathElement(best_aligned_index, intersection_points, intersection_indices, end_points, end_indices))
                    
                    # self.fil_length_pixels += self.filament.branch_properties["length"][best_aligned_index].value
                    # branch_points_array = np.asarray(self.filament.branch_pts(True)[best_aligned_index])
                    # self.long_path_binary[branch_points_array[:,0], branch_points_array[:,1]] = 1
                
                    # if not (self.AddPointsToLongPathInOrder(best_aligned_index)):
                    #     return False
                    
                    # image = cv2.resize(np.rint((self.long_path_binary * 255 + self.contour)/2).astype('uint8'), None, fy=0.5, fx=0.5)
                    # cv2.imshow("Evolution", image)
                    # cv2.waitKey(0)
                    
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
                print("All feasible SLP potentials weeded out")
                self.standard_length_point = None
                self.slp_near_endpoint = True
        else:
            print("No SLP potentials found")
            self.standard_length_point = None
            self.slp_near_endpoint = True

    def AddBoundaryDistances(self):
        # Get the points that are "pulled back" from the origin point, from which the added distance lines will stem
        # These lines will run through the pullback point, through the origin point, and intersect with the contour of the blob
        pullback_pts = []

        # For the head point, use a circle of radius 20 pixels to find a point further back on the skeleton
        circle_mask = np.zeros(self.dimensions)
        circle_radius = 20
        for x in range(self.head_point[1] - circle_radius, self.head_point[1] + circle_radius + 1):
            for y in range(self.head_point[0] - circle_radius, self.head_point[0] + circle_radius + 1):
                if x <= self.dimensions[1] - 1 and x >= 0:
                    if y <= self.dimensions[0] - 1 and y >= 0:
                        ref_val = ((x-self.head_point[1])**2 + (y-self.head_point[0])**2)**(0.5)
                        if ref_val >= circle_radius - 0.7 and ref_val <= circle_radius + 0.7:
                            circle_mask[y][x] = 1
        
        longpath_bool_array = PointUtils.ContainsMutualPoints(np.argwhere(circle_mask), self.long_path_pixel_coords, return_array=True)
        
        try:
            head_pullback_pt = self.long_path_pixel_coords[np.argwhere(longpath_bool_array)[0], :]
        except IndexError:
            print("error1")
            # image = cv2.resize(np.rint((circle_mask + self.long_path_binary)/2 * 255).astype('uint8'), None, fy=0.5, fx=0.5)
            # cv2.imshow("Circle Mask", image)
            # cv2.waitKey(0)
            return False
            
        pullback_pts.append(head_pullback_pt)
        print("Head pullback point at {0}".format(head_pullback_pt))

        # For the tail point, use the standard length point if it's not within a 20 pixel radius of the end point
        # otherwise, also use a circle mask
        if (not self.slp_near_endpoint):
            pullback_pts.append(self.standard_length_point)
            print("Using standard length point for tail pullback, at {0}".format(self.standard_length_point))
            
            # We have to remove the added length if this case occurs
            print("Trimming the branches")
            if not self.RemoveAdditionalTailLength(self.standard_length_point):
                return False
        else:
            circle_mask = np.zeros(self.dimensions)
            circle_radius = 20
            for x in range(self.tail_point[1] - circle_radius, self.tail_point[1] + circle_radius + 1):
                for y in range(self.tail_point[0] - circle_radius, self.tail_point[0] + circle_radius + 1):
                    if x <= self.dimensions[1] - 1 and x >= 0:
                        if y <= self.dimensions[0] - 1 and y >= 0:
                            ref_val = ((x-self.tail_point[1])**2 + (y-self.tail_point[0])**2)**(0.5)
                            if ref_val >= circle_radius - 0.7 and ref_val <= circle_radius + 0.7:
                                circle_mask[y][x] = 1

            longpath_bool_array = PointUtils.ContainsMutualPoints(np.argwhere(circle_mask), self.long_path_pixel_coords, return_array=True)
            try:
                pullback_pt = self.long_path_pixel_coords[np.argwhere(longpath_bool_array)[0], :]
                pullback_pts.extend(list(map(tuple, pullback_pt)))
                print("Using circle radius point for tail pullback, at {0}".format(pullback_pt))
            except IndexError:
                # kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))) + self.end_point[0] - (3,3))
                # circle_mask[kernel[:,0], kernel[:,1]] = 1
                print("error2")
                
                # image = cv2.resize(np.rint(combined_array * 255).astype('uint8'), None, fy=0.3, fx=0.3)
                # cv2.imshow("Failed overlap array", image)
                # cv2.waitKey(0)
                

        # Run the assessment for each pullback/endpoint pair
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
                    self.long_path_binary[end_pt[0]:closest_boundary_point[0]+step_y:step_y, end_pt[1]] = \
                    line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y, end_pt[1]]
                elif end_pt[0] == closest_boundary_point[0]:
                    self.long_path_binary[end_pt[0], end_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                    line_mask[end_pt[0], end_pt[1]:closest_boundary_point[1]+step_x:step_x]
                else:
                    self.long_path_binary[end_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                        end_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                    line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                            end_pt[1]:closest_boundary_point[1]+step_x:step_x]
                
                distance = np.linalg.norm(np.asarray(end_pt) - np.asarray(closest_boundary_point))
                print("Added head line to path, measuring {0:.2f} pixels".format(distance))
                self.long_path_binary = PointUtils.AddThickBinaryDots(self.long_path_binary, closest_boundary_point, size=(11,11))
                
                # current_array = np.rint(np.add(self.long_path_binary * 255, self.contour)/2).astype('uint8')
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
                        self.long_path_binary[pullback_pt[0]:closest_boundary_point[0]+step_y:step_y, pullback_pt[1]] = \
                        line_mask[pullback_pt[0]:closest_boundary_point[0]+step_y:step_y, pullback_pt[1]]
                    elif pullback_pt[0] == closest_boundary_point[0]:
                        self.long_path_binary[pullback_pt[0], pullback_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                        line_mask[pullback_pt[0], pullback_pt[1]:closest_boundary_point[1]+step_x:step_x]
                    else:
                        self.long_path_binary[pullback_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                            pullback_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                        line_mask[pullback_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                pullback_pt[1]:closest_boundary_point[1]+step_x:step_x]
                    
                    distance = np.linalg.norm(np.asarray(pullback_pt) - np.asarray(closest_boundary_point))
                    
                    # current_array = np.rint(np.add(self.long_path_binary * 255, self.contour)/2).astype('uint8')
                    # image = cv2.resize(current_array, None, fy=0.5, fx=0.5)
                    # cv2.imshow("tail added; SLP", image)
                    # cv2.waitKey(0)

                else:
                    print("Closest boundary point to TAIL endpoint {0}, is {1}".format(end_pt, closest_boundary_point))
                    step_y = 1 if end_pt[0]<=closest_boundary_point[0] else -1
                    step_x = 1 if end_pt[1]<=closest_boundary_point[1] else -1
                    if end_pt[1] == closest_boundary_point[1]:
                        self.long_path_binary[end_pt[0]:closest_boundary_point[0]+step_y:step_y, end_pt[1]] = \
                        line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y, end_pt[1]]
                    elif end_pt[0] == closest_boundary_point[0]:
                        self.long_path_binary[end_pt[0], end_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                        line_mask[end_pt[0], end_pt[1]:closest_boundary_point[1]+step_x:step_x]
                    else:
                        self.long_path_binary[end_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                            end_pt[1]:closest_boundary_point[1]+step_x:step_x] = \
                        line_mask[end_pt[0]:closest_boundary_point[0]+step_y:step_y,\
                                end_pt[1]:closest_boundary_point[1]+step_x:step_x]
                    
                    distance = np.linalg.norm(np.asarray(end_pt) - np.asarray(closest_boundary_point))
                    
                    # current_array = np.rint(np.add(self.long_path_binary * 255, self.contour)/2).astype('uint8')
                    # image = cv2.resize(current_array, None, fy=0.5, fx=0.5)
                    # cv2.imshow("tail added; pullback", image)
                    # cv2.waitKey(0)
                
                print("Added tail line to path, measuring {0:.2f} pixels".format(distance))
                self.long_path_binary = PointUtils.AddThickBinaryDots(self.long_path_binary, closest_boundary_point, size=(11,11))

            self.fil_length_pixels += distance
        
        return True
        
    def RemoveAdditionalTailLength(self, pullback_pt):
        ### rewrite this to make use of ELEMENTS
        ## maybe put this in SLP method, or reference it there instead of contour distances
        
        
        
        
        # Figure out which branch of the longest path contains the pullback point
        pullback_branch_index = self.GetBranchFromPoint(pullback_pt)[0]
        print("Branch {0} is the pullback branch".format(pullback_branch_index))
        
        if pullback_branch_index is not None:
            pullback_index_LP_array = self.longest_path_branch_indices.index(pullback_branch_index)
            print("This branch is at position {0} in the LP branches array (0-indexed). There are {1} branches in this array".format(pullback_index_LP_array, len(self.longest_path_branch_indices)))
            
            # Iteratively remove branches in reverse order up to that branch
            for i in range(len(self.longest_path_branch_indices)-1, pullback_index_LP_array, -1):
                print("Current branch indices: {0}".format(self.longest_path_branch_indices))
                print("Removing branch {0} w/length {1} pixels".format(self.longest_path_branch_indices[i], self.filament.branch_properties["length"][self.longest_path_branch_indices[i]].value))
                
                # base_length = self.filament.branch_properties["length"][i].value
                # self.fil_length_pixels += base_length
                # unit_length = base_length / len(base_branch_points)
                
                self.fil_length_pixels -= self.filament.branch_properties["length"][self.longest_path_branch_indices[i]].value
                
                index_array = np.asarray(self.filament.branch_pts(True)[self.longest_path_branch_indices[i]])
                self.long_path_binary[index_array[:,0], index_array[:,1]] = 0
                
                del self.longest_path_branch_indices[i]
                print("New branch indices: {0}".format(self.longest_path_branch_indices))
                            
            # Remove all points in the pullback branch up until the pullback point
            # The branch may not be ordered from head to tail, use the ordered long_path_pixel_coords to verify
            branch_points = self.filament.branch_pts(True)[pullback_branch_index]
            pullback_point_index = np.argwhere(np.equal(np.asarray(self.long_path_pixel_coords), pullback_pt).all(axis=1))[0][0]
            self.long_path_pixel_coords = self.long_path_pixel_coords[:pullback_point_index + 1]
            
            if PointUtils.ContainsMutualPoints(branch_points[0], np.asarray(self.long_path_pixel_coords)):
                # The first point is still in, so the branch is oriented head --> tail
                print("{0} is still in long path coords array, head--> tail".format(branch_points[0]))
                extraneous_points_array = np.asarray(branch_points[np.argwhere(np.equal(branch_points, pullback_pt).all(axis=1))[0][0]+1:])
            else:
                # The first point is not in, so the branch is oriented tail --> head
                print("{0} is NOT in long path coords array anymore, tail --> head".format(branch_points[0]))
                extraneous_points_array = np.asarray(branch_points[:np.argwhere(np.equal(branch_points, pullback_pt).all(axis=1))[0][0]])
            
            self.fil_length_pixels -= np.shape(extraneous_points_array)[0]
            self.long_path_binary[extraneous_points_array[:,0], extraneous_points_array[:,1]] = 0
            self.long_path_binary = PointUtils.AddThickBinaryDots(self.long_path_binary, self.standard_length_point)
            
            # current_array = np.rint(np.add(self.long_path_binary * 255, self.contour)/2).astype('uint8')
            # image = cv2.resize(current_array, None, fy=0.5, fx=0.5)
            # cv2.imshow("Post-trim", image)
            # cv2.waitKey(0)
            
            return True
        else:
            print("Could not identify tail pullback point base branch")
            return False
        
    # def AddPointsToLongPathInOrder(self, new_branch_index):
    #     print("Adding {0} points to ordered long path. Current length before addition: {1}".format(len(self.filament.branch_pts(True)[new_branch_index]), len(self.long_path_pixel_coords)))
        
    #     if np.asarray(self.long_path_pixel_coords).size == 0:
    #         print("coords path is empty, initializing")
    #         self.long_path_pixel_coords = [point for point in self.filament.branch_pts(True)[new_branch_index]]
            
    #         if not PointUtils.PointInNeighborhood(self.head_point, self.long_path_pixel_coords[0]):
    #             print("head point at other end, reversing order")
    #             self.long_path_pixel_coords.reverse()       
            
    #         return True
    #     else:
    #         # Get the shared intersection point
    #         intersection = None
    #         for intersec_pt in self.filament.intersec_pts:
    #             contending_intersec_pts = intersec_pt if isinstance(intersec_pt, list) else [intersec_pt]
    #             for point in contending_intersec_pts:
    #                 kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))) + point - (3,3))
                    
    #                 branch_mutual_points = PointUtils.ContainsMutualPoints(kernel, self.filament.branch_pts(True)[new_branch_index])
    #                 longpath_mutual_points = PointUtils.ContainsMutualPoints(kernel, np.asarray(self.long_path_pixel_coords))
                    
    #                 # if the intersection point exists both on the new branch and the current long path, it's the right one
    #                 if branch_mutual_points and longpath_mutual_points:
    #                     intersection = point
    #                     print("Found the shared intersection point at: {0}".format(point))
    #                     break
                    
    #             if intersection is not None:
    #                 break
            
    #         # Remove length up until this point
    #         if intersection is None:
    #             print("Could not find the shared intersection point")
    #             return False
    #         else:
    #             if self.long_path_binary[intersection] == 0:
    #                 self.long_path_binary[intersection] = 1
    #                 self.fil_length_pixels += 1
                                
    #             intersection_bool_array = PointUtils.ContainsMutualPoints(kernel, np.asarray(self.long_path_pixel_coords), return_array=True)
    #             intersection_index_on_longpath = np.where(intersection_bool_array == True)[0][-1]
                
    #             # If the intersection point is the last point on the current long path, do nothing
    #             if intersection_index_on_longpath != len(self.long_path_pixel_coords) - 1:
    #                 dist_to_remove = len(self.long_path_pixel_coords[intersection_index_on_longpath + 1:])
    #                 self.fil_length_pixels -= dist_to_remove
    #                 print("Trimming current long path by {0} pixels".format(dist_to_remove))
                    
    #                 self.long_path_pixel_coords = self.long_path_pixel_coords[:intersection_index_on_longpath + 1]
    #             else:
    #                 print("Intersection point is the last point on the current long path")
            
    #         # Find on which side the shared intersection point is. Point may not be exactly on the branch, use a 5x5 kernel
    #         base_start_point = self.long_path_pixel_coords[0]
    #         base_end_point = self.long_path_pixel_coords[-1]
    #         new_branch_start_point = self.filament.branch_pts(True)[new_branch_index][0]
    #         new_branch_end_point = self.filament.branch_pts(True)[new_branch_index][-1]
            
    #         base_intersection_near_start = PointUtils.PointInNeighborhood(intersection, base_start_point)
    #         new_branch_intersection_near_start = PointUtils.PointInNeighborhood(intersection, new_branch_start_point)
            
    #         print("start_base: {0}; end_base: {1}; start_branch: {2}; end_branch: {3}".format(base_start_point, base_end_point, new_branch_start_point, new_branch_end_point))

    #         # Add to the longpath coords, subject to the location of the shared intersection point
    #         if base_intersection_near_start and new_branch_intersection_near_start:
    #             # Start-start connection, reverse new branch order and insert at front of list
    #             print("start-start")
    #             for point in self.filament.branch_pts(True)[new_branch_index]:
    #                 self.long_path_pixel_coords.insert(0, point)
    #         elif base_intersection_near_start and not new_branch_intersection_near_start:
    #             # Start-end connection, insert at front of list
    #             print("start-end")
    #             try:
    #                 if isinstance(self.long_path_pixel_coords, list):
    #                     self.long_path_pixel_coords = self.filament.branch_pts(True)[new_branch_index].tolist() + self.long_path_pixel_coords
    #                 else:
    #                     self.long_path_pixel_coords = np.vstack([self.filament.branch_pts(True)[new_branch_index], self.long_path_pixel_coords])
    #             except ValueError:
    #                 print(type(self.filament.branch_pts(True)[new_branch_index]))
    #                 print(type(self.long_path_pixel_coords))
    #         elif not base_intersection_near_start and new_branch_intersection_near_start:       
    #             # End-start connection, append to back of list
    #             print("end-start")
    #             self.long_path_pixel_coords.extend(self.filament.branch_pts(True)[new_branch_index])
    #         elif not base_intersection_near_start and not new_branch_intersection_near_start:       
    #             # End-end connection, reverse new branch order and append to back of list
    #             print("end-end")
    #             for i in range(len(self.filament.branch_pts(True)[new_branch_index])-1, -1, -1):
    #                 self.long_path_pixel_coords.append(self.filament.branch_pts(True)[new_branch_index][i])
    #         else:
    #             print("Connection of added branch unclear. Aborting instance.")
    #             return False
        
    #     print("New length: {0}".format(len(self.long_path_pixel_coords)))
    #     return True

    def GetBranchFromPoint(self, point, size=None):
        # Supply a point (x, y), and get the branch that point belongs to if it exists
        branches = []
        if size is None:
            for i in range(self.number_of_branches):
                if PointUtils.ContainsMutualPoints(np.asarray(point), self.filament.branch_pts(True)[i]):
                    return [i]
        else:
            kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)) + point - ((np.asarray(size) - 1) / 2).astype('uint8'))
            for i in range(self.number_of_branches):
                if PointUtils.ContainsMutualPoints(kernel, self.filament.branch_pts(True)[i]):
                    branches.append(i)
        
        return branches
            
    def GetBranchIntersections(self, branch_index, also_indices=False):
        # Get the intersection point(s) on the branch
        intersec_pts_on_branch = []
        intersec_pts_on_branch_indices = []

        for i in range(len(self.filament.intersec_pts)):
            intersec_pts = self.filament.intersec_pts[i] if isinstance(self.filament.intersec_pts[i], list) else [self.filament.intersec_pts[i]]
            for point in intersec_pts:
                kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) + point - (2,2))
                branch_mutual_points = PointUtils.ContainsMutualPoints(kernel, self.filament.branch_pts(True)[branch_index])

                if branch_mutual_points:
                    intersec_pts_on_branch.append(point)
                    intersec_pts_on_branch_indices.append(i)
        
        print("Branch at index: {0} contains the following intersection points: {1}".format(branch_index, intersec_pts_on_branch))
        if not also_indices:
            return intersec_pts_on_branch
        else:
            return intersec_pts_on_branch, intersec_pts_on_branch_indices

    def GetBranchEndPoints(self, branch_index, also_indices=False):
        # Get the intersection point(s) on the branch
        end_pts_on_branch = []
        end_pts_on_branch_indices = []
        
        for i in range(len(self.filament.end_pts)):
            end_pt = self.filament.end_pts[i]
            if PointUtils.ContainsMutualPoints(np.asarray(end_pt), self.filament.branch_pts(True)[branch_index]):
                end_pts_on_branch.append(end_pt)
                end_pts_on_branch_indices.append(i)
                        
        print("Branch at index: {0} contains the following end points: {1}".format(branch_index, end_pts_on_branch))
        if not also_indices:
            return end_pts_on_branch
        else:
            return end_pts_on_branch, end_pts_on_branch_indices

    def FindHeadPoint(self):
        # Find the starting point: the skeleton end point in the head
        contending_pt = None
        for i in range(len(self.filament.end_pts)):

            # The entry may be a list of closely located points, just take the first one if so
            if isinstance(self.filament.end_pts[i], list):
                contending_pt = self.filament.end_pts[i][0]
            else:
                contending_pt = self.filament.end_pts[i]

            if self.head_point is None:
                self.head_point = contending_pt
            else:
                print("head_point dist: {0}; contending_pt dist: {1}".format(self.distance_transform[self.head_point], self.distance_transform[contending_pt]))
                if self.distance_transform[contending_pt] > self.distance_transform[self.head_point]:
                    self.head_point = contending_pt

        print("Head point: {0}".format(self.head_point))
        
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
                    
    def AddBranch(self, base_element):
        base_branch_points = self.filament.branch_pts(True)[base_element.branch_index]
        print("Branch contains {0} points".format(np.shape(base_branch_points)[0]))
        
        head_kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) + base_element.head_point - (2,2))
        head_bool_array = PointUtils.ContainsMutualPoints(head_kernel, base_branch_points, return_array=True)
        if not any(head_bool_array):
            return False
        head_index = np.argwhere(head_bool_array)[0][0]
        
        tail_kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) + base_element.tail_point - (2,2))
        tail_bool_array = PointUtils.ContainsMutualPoints(tail_kernel, base_branch_points, return_array=True)
        if not any(tail_bool_array):
            return False
        tail_index = np.argwhere(tail_bool_array)[0][-1]
        
        print("Head index: {0}; tail index: {1}".format(head_index, tail_index))
        if head_index > tail_index:
            print("Branch orientation is from tail to head, reversing")
            base_branch_points = np.flip(base_branch_points, axis=0)
        else:
            print("Branch is correctly oriented")
        
        # Then determine if we need to remove length from the base branch
        base_length = self.filament.branch_properties["length"][base_element.branch_index].value
        self.fil_length_pixels += base_length
        unit_length = base_length / len(base_branch_points)
        
        if not PointUtils.PointInNeighborhood(base_element.head_point, base_branch_points[0]):
            # The head point occurs somewhere mid-branch, we need to trim up until that point
            trimmed_length = head_index * unit_length
            self.fil_length_pixels -= trimmed_length
            base_branch_points = base_branch_points[head_index:, :]
            print("Head index is mid-branch, trimming {0:.2f} pixels".format(trimmed_length))
        else:
            print("Head point is at the beginning of the list")
            
        if not PointUtils.PointInNeighborhood(base_element.tail_point, base_branch_points[-1]):
            # The tail point occurs somewhere mid-branch, we need to trim from that point on
            trimmed_length = len(base_branch_points) - tail_index * unit_length
            self.fil_length_pixels -= trimmed_length
            base_branch_points = base_branch_points[:tail_index+1, :]
            print("Tail index is mid-branch, trimming {0:.2f} pixels".format(trimmed_length))
        else:
            print("Tail point is at the end of the list")
            
        # Record the length and save the coords
        self.long_path_pixel_coords.extend(list(map(tuple, base_branch_points)))
        
        return True
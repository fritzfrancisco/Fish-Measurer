import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as u
from skimage import img_as_bool
from skimage.morphology import medial_axis, binary_closing, binary_opening
from scipy.signal import find_peaks
import os

class Measurement():
    def __init__(self, process_id, raw_frame, binarized_frame, outputfolder):
        # Direct attributions
        self.process_id = process_id
        self.raw_frame = raw_frame
        self.binarized_frame = binarized_frame
        self.outputfolder = outputfolder

        # Image arrays
        self.contour = None
        self.skeleton = None
        self.distance_transform = None
        
        self.dimensions = np.shape(self.binarized_frame)
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
        self.path_endpoints = []
        self.covered_intersec_pt_indices = []
        self.covered_branch_indices = []
        self.number_of_branches = None
        self.long_path_pixel_coords = []
        
        # Standard length point
        self.standard_length_point = None
        self.slp_near_endpoint = None


        ## -- START -- ##
        # Apply morphological operations (image processing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(binarized_frame, cv2.MORPH_OPEN, kernel)
        self.contour = cv2.morphologyEx(cv2.erode(opening,kernel,iterations = 1), cv2.MORPH_GRADIENT, kernel)
        # self.contour = cv2.addWeighted(opening,0.5,cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel) ,0.5,0)

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

        # Find the starting point: the skeleton end point in the head
        head_pt = None
        for i in range(len(self.filament.end_pts)):
            contending_pt = None

            # The entry may be a list of closely located points, just take the first one if so
            if isinstance(self.filament.end_pts[i], list):
                contending_pt = self.filament.end_pts[i][0]
            else:
                contending_pt = self.filament.end_pts[i]

            if head_pt is None:
                head_pt = contending_pt
            else:
                print("head_pt dist: {0}; contending_pt dist: {1}".format(self.distance_transform[head_pt], self.distance_transform[contending_pt]))
                if self.distance_transform[contending_pt] > self.distance_transform[head_pt]:
                    head_pt = contending_pt

        print("Head point: {0}".format(head_pt))
        self.path_endpoints.append(head_pt)

        # Fetch the branch that contains this head point
        base_branch_index = None
        for i in range(self.number_of_branches):
            if any(np.equal(self.filament.branch_pts(True)[i], [head_pt[0],head_pt[1]]).all(axis=1)):
                self.longest_path_branch_indices.append(i)
                self.covered_branch_indices.append(i)
                base_branch_index = i
                print("starting branch is branch {0}".format(i))
                
                self.fil_length_pixels += self.filament.branch_properties["length"][i].value
                branch_points_array = np.asarray(self.filament.branch_pts(True)[i])
                self.long_path_binary[branch_points_array[:,0], branch_points_array[:,1]] = 1
                
                # image = cv2.resize(np.rint((self.long_path_binary * 255 + self.contour)/2).astype('uint8'), None, fy=0.5, fx=0.5)
                # cv2.imshow("Evolution", image)
                # cv2.waitKey(0)
                
                # image = cv2.resize(np.rint((self.tempos * 255 + self.contour)/2).astype('uint8'), None, fy=0.5, fx=0.5)
                # path = os.path.join(self.outputfolder, "evolution_{0}{1}".format(i, ".jpg"))
                # cv2.imwrite(path, image)
                
                break
            else:
                print("starting branch is NOT branch {0}".format(i))

        if not (self.AddPointsToLongPathInOrder(base_branch_index)):
            return False
        
        if not self.RecursiveBranching(base_branch_index):
            return False
    
        for point in self.path_endpoints:
            self.AddThickDotLP(point)
            self.long_path_binary[point] = 1
        
        # Get proper ordering of the longpath coords
        if self.PointInNeighborhood(self.path_endpoints[0], self.long_path_pixel_coords[0]):
            print("Head is within 5x5 kernel space at front of list")
        else:
            if self.PointInNeighborhood(self.path_endpoints[0], self.long_path_pixel_coords[-1]):
                print("Head is within 5x5 kernel space at back of list. Reversing.")
                self.long_path_pixel_coords.reverse()
            else:
                print("Where the hell is the head point? Checking full list")
                if any(np.equal(self.long_path_pixel_coords, self.path_endpoints[0]).all(axis=1)):
                    print("Found in list at index {0}".format(np.where(np.equal(self.long_path_pixel_coords, self.path_endpoints[0]).all(axis=1))))
                else:
                    print("head point not found, genuinely have no idea wassup")
                    return False

        # Fill in the image matrices & grab combined length
        # for branch_index in self.longest_path_branch_indices:
        #     self.fil_length_pixels += self.filament.branch_properties["length"][branch_index].value
            
        #     branch_points_array = np.asarray(self.filament.branch_pts(True)[branch_index])
        #     self.long_path_binary[branch_points_array[:,0], branch_points_array[:,1]] = 1
            
        # for y, x in self.path_endpoints:
        #     self.long_path_binary[y][x] = 1
        
        # Ensure that the long path operation was successful any meaningful by checking if
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

            found_point = False
            for end_pt in [point for point in self.filament.end_pts if point not in self.path_endpoints]:
                if any(np.equal(self.filament.branch_pts(True)[base_branch_index], [end_pt[0],end_pt[1]]).all(axis=1)):
                    print("end point {0} found in branch {1}".format(end_pt, base_branch_index))
                    self.path_endpoints.append(end_pt)
                    found_point = True

            if not found_point:
                print("Did not find the endpoint of the terminal branch")
                return False
            else:
                return True

        else:
            # Get the intersection point(s) on the base branch that haven't been covered
            intersec_pts_on_basebranch_indices = []
            contending_intersec_pt_indices = [index for index in range(len(self.filament.intersec_pts)) if index not in self.covered_intersec_pt_indices]
            print("Covered intersection points: {0}; Remaining: {1}".format(self.covered_intersec_pt_indices, contending_intersec_pt_indices))

            for i in contending_intersec_pt_indices:
                contending_intersec_pts = self.filament.intersec_pts[i] if isinstance(self.filament.intersec_pts[i], list) else [self.filament.intersec_pts[i]]
                print("Intersection point {0}: {1}".format(i, contending_intersec_pts))
                for point in contending_intersec_pts:
                    # Each intersection point belongs to only ONE branch, and is not shared
                    # Use a 5x5 kernel centered on the intersection point to determine shared ownshership
                    kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) + point - (2,2))
                    branch_bool_array = (kernel[:, None] == self.filament.branch_pts(True)[base_branch_index]).all(axis=2).any(axis=0)

                    if any(branch_bool_array):
                        intersec_pts_on_basebranch_indices.append(i)
                        if i not in self.covered_intersec_pt_indices:
                            self.covered_intersec_pt_indices.append(i)
                        print("Intersection point exists on base branch at index {0}".format(i))
                        break
                    else:
                        print("Intersection point at index {0} doesn't exist on base branch".format(i))

            if not intersec_pts_on_basebranch_indices:
                # The end of the branch must be an endpoint, we're done
                print("We're done-so's, no intersection points found on the base branch")

                found_point = False
                for end_pt in [point for point in self.filament.end_pts if point not in self.path_endpoints]:
                    if any(np.equal(self.filament.branch_pts(True)[base_branch_index], [end_pt[0],end_pt[1]]).all(axis=1)):
                        print("end point {0} found in branch {1}".format(end_pt, base_branch_index))
                        self.path_endpoints.append(end_pt)
                        found_point = True

                if not found_point:
                    print("Did not find the endpoint of the terminal branch")
                    return False
                else:
                    return True
            else:
                # Find which other branches also share this intersection point(s)
                connected_branch_indices = []
                for contending_branch_index in [i for i in range(self.number_of_branches) if i not in self.covered_branch_indices]:
                    for index in intersec_pts_on_basebranch_indices:
                        contending_intersec_pts = self.filament.intersec_pts[index] if isinstance(self.filament.intersec_pts[index], list) else [self.filament.intersec_pts[index]]
                        for point in contending_intersec_pts:
                            # Again, use a kernel to evaluate the neighborhood
                            kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))) + point - (2,2))
                            branch_bool_array = (kernel[:, None] == self.filament.branch_pts(True)[contending_branch_index]).all(axis=2).any(axis=0)

                            if any(branch_bool_array):
                                if contending_branch_index not in connected_branch_indices:
                                    connected_branch_indices.append(contending_branch_index)
                                    
                                if contending_branch_index not in self.covered_branch_indices:
                                    self.covered_branch_indices.append(contending_branch_index)
                                    
                                break

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
                    
                    self.fil_length_pixels += self.filament.branch_properties["length"][best_aligned_index].value
                    branch_points_array = np.asarray(self.filament.branch_pts(True)[best_aligned_index])
                    self.long_path_binary[branch_points_array[:,0], branch_points_array[:,1]] = 1
                
                    if not (self.AddPointsToLongPathInOrder(best_aligned_index)):
                        return False
                    
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
        lp_pixel_array = np.asarray(self.long_path_pixel_coords)
        longpath_distance_array = self.distance_transform[lp_pixel_array[:,0], lp_pixel_array[:,1]]
        average_distance = np.average(longpath_distance_array)
        local_minima_indices, _ = find_peaks(-longpath_distance_array, height=-average_distance, prominence=1)
        
        print("Possible SLPs at indices: {0}".format(local_minima_indices))
        print("Corresponding values: {0}".format(lp_pixel_array[local_minima_indices]))
            
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
                print("Standard length point found at index {2} of {3}: {0}; Tailpoint: {1}".format(self.standard_length_point, self.path_endpoints[1], global_index, len(self.long_path_pixel_coords)))
                
                # Check whether the SLP meaningfully exists, or whether it's basically the tail endpoint
                self.slp_near_endpoint = False
                if self.PointInNeighborhood(self.standard_length_point, self.path_endpoints[1], size=(41,41)):
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
        for x in range(self.path_endpoints[0][1] - circle_radius, self.path_endpoints[0][1] + circle_radius + 1):
            for y in range(self.path_endpoints[0][0] - circle_radius, self.path_endpoints[0][0] + circle_radius + 1):
                if x <= self.dimensions[1] - 1 and x >= 0:
                    if y <= self.dimensions[0] - 1 and y >= 0:
                        ref_val = ((x-self.path_endpoints[0][1])**2 + (y-self.path_endpoints[0][0])**2)**(0.5)
                        if ref_val >= circle_radius - 0.7 and ref_val <= circle_radius + 0.7:
                            circle_mask[y][x] = 1

        combined_array = np.add(circle_mask, self.long_path_binary)
        try:
            head_pullback_pt = list(zip(*np.where(combined_array > 1.5)))[0]
        except IndexError:
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
            for x in range(self.path_endpoints[1][1] - circle_radius, self.path_endpoints[1][1] + circle_radius + 1):
                for y in range(self.path_endpoints[1][0] - circle_radius, self.path_endpoints[1][0] + circle_radius + 1):
                    if x <= self.dimensions[1] - 1 and x >= 0:
                        if y <= self.dimensions[0] - 1 and y >= 0:
                            ref_val = ((x-self.path_endpoints[1][1])**2 + (y-self.path_endpoints[1][0])**2)**(0.5)
                            if ref_val >= circle_radius - 0.7 and ref_val <= circle_radius + 0.7:
                                circle_mask[y][x] = 1

            combined_array = np.add(circle_mask, self.long_path_binary)
            try:
                pullback_pt = list(zip(*np.where(combined_array > 1.5)))[0]
                pullback_pts.append(pullback_pt)
                print("Using circle radius point for tail pullback, at {0}".format(pullback_pt))
            except IndexError:
                kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))) + self.end_point[0] - (3,3))
                combined_array[kernel[:,0], kernel[:,1]] = 1
                
                # image = cv2.resize(np.rint(combined_array * 255).astype('uint8'), None, fy=0.3, fx=0.3)
                # cv2.imshow("Failed overlap array", image)
                # cv2.waitKey(0)
                

        # Run the assessment for each pullback/endpoint pair
        for i in range(2):
            # First create the line mask through the pullback/end points
            line_mask = np.zeros(self.dimensions)
            pullback_pt = pullback_pts[i]
            end_pt = self.path_endpoints[i]

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
                self.AddThickDotLP(closest_boundary_point, size=(11,11))
                
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
                self.AddThickDotLP(closest_boundary_point, size=(11,11))

            self.fil_length_pixels += distance
        
        return True
        
    def GetBranchFromPoint(self, point):
        # Supply a point (x, y), and get the branch that point belongs to if it exists
        for i in range(self.number_of_branches):
            if any(np.equal(self.filament.branch_pts(True)[i], point).all(axis=1)):
                return i
            
        return None
    
    def RemoveAdditionalTailLength(self, pullback_pt):
        # Figure out which branch of the longest path contains the pullback point
        pullback_branch_index = self.GetBranchFromPoint(pullback_pt)
        print("Branch {0} is the pullback branch".format(pullback_branch_index))
        
        if pullback_branch_index is not None:
            pullback_index_LP_array = self.longest_path_branch_indices.index(pullback_branch_index)
            print("This branch is at position {0} in the LP branches array (0-indexed). There are {1} branches in this array".format(pullback_index_LP_array, len(self.longest_path_branch_indices)))
            
            # Iteratively remove branches in reverse order up to that branch
            for i in range(len(self.longest_path_branch_indices)-1, pullback_index_LP_array, -1):
                print("Current branch indices: {0}".format(self.longest_path_branch_indices))
                print("Removing branch {0} w/length {1} pixels".format(self.longest_path_branch_indices[i], self.filament.branch_properties["length"][self.longest_path_branch_indices[i]].value))
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
            
            if any(np.equal(self.long_path_pixel_coords, branch_points[0]).all(axis=1)):
                # The first point is still in, so the branch is oriented head --> tail
                print("{0} is still in long path coords array, head--> tail".format(branch_points[0]))
                extraneous_points_array = np.asarray(branch_points[np.argwhere(np.equal(branch_points, pullback_pt).all(axis=1))[0][0]+1:])
            else:
                # The first point is not in, so the branch is oriented tail --> head
                print("{0} is NOT in long path coords array anymore, tail --> head".format(branch_points[0]))
                extraneous_points_array = np.asarray(branch_points[:np.argwhere(np.equal(branch_points, pullback_pt).all(axis=1))[0][0]])
            
            self.fil_length_pixels -= np.shape(extraneous_points_array)[0]
            self.long_path_binary[extraneous_points_array[:,0], extraneous_points_array[:,1]] = 0
            self.AddThickDotLP(self.standard_length_point)
            
            # current_array = np.rint(np.add(self.long_path_binary * 255, self.contour)/2).astype('uint8')
            # image = cv2.resize(current_array, None, fy=0.5, fx=0.5)
            # cv2.imshow("Post-trim", image)
            # cv2.waitKey(0)
            
            return True
        else:
            print("Could not identify tail pullback point base branch")
            return False
    
    def PointInNeighborhood(self, point1, point2, size=(5,5)):
        # Where point1 and point2 are (y, x) tuples
        # Get the circlular kernel of radius 20 and offset to endpoint coords
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)
        neighboring_indices = np.argwhere(kernel) + point2 - ((np.asarray(size) - 1) / 2)
        
        # Check if the slp is in neighborhood
        if any(np.equal(neighboring_indices, [point1[0],point1[1]]).all(axis=1)):
            return True
        else:
            return False
    
    def AddPointsToLongPathInOrder(self, new_branch_index):
        print("Adding {0} points to ordered long path. Current length before addition: {1}".format(len(self.filament.branch_pts(True)[new_branch_index]), len(self.long_path_pixel_coords)))
        
        if np.asarray(self.long_path_pixel_coords).size == 0:
            print("coords path is empty, initializing")
            self.long_path_pixel_coords = [point for point in self.filament.branch_pts(True)[new_branch_index]]
            
            if not self.PointInNeighborhood(self.path_endpoints[0], self.long_path_pixel_coords[0]):
                print("head point at other end, reversing order")
                self.long_path_pixel_coords.reverse()       
            
            return True
        else:
            # Get the shared intersection point
            intersection = None
            for intersec_pt in self.filament.intersec_pts:
                contending_intersec_pts = intersec_pt if isinstance(intersec_pt, list) else [intersec_pt]
                for point in contending_intersec_pts:
                    kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))) + point - (3,3))
                    
                    branch_bool_array = (kernel[:, None] == self.filament.branch_pts(True)[new_branch_index]).all(axis=2).any(axis=0)
                    longpath_bool_array = (kernel[:, None] == np.asarray(self.long_path_pixel_coords)).all(axis=2).any(axis=0)

                    # if the intersection point exists both on the new branch and the current long path, it's the right one
                    if any(branch_bool_array) and any(longpath_bool_array):
                        intersection = point
                        print("Found the shared intersection point at: {0}".format(point))
                        break
                    
                if intersection is not None:
                    break
            
            # Remove length up until this point
            if intersection is None:
                print("Could not find the shared intersection point")
                return False
            else:
                if self.long_path_binary[intersection] == 0:
                    self.long_path_binary[intersection] = 1
                    self.fil_length_pixels += 1
                
                intersection_bool_array = (kernel[:, None] == np.asarray(self.long_path_pixel_coords)).all(axis=2).any(axis=0)
                intersection_index_on_longpath = np.where(intersection_bool_array == True)[0][-1]
                
                # If the intersection point is the last point on the current long path, do nothing
                if intersection_index_on_longpath != len(self.long_path_pixel_coords) - 1:
                    dist_to_remove = len(self.long_path_pixel_coords[intersection_index_on_longpath + 1:])
                    self.fil_length_pixels -= dist_to_remove
                    print("Trimming current long path by {0} pixels".format(dist_to_remove))
                    
                    self.long_path_pixel_coords = self.long_path_pixel_coords[:intersection_index_on_longpath + 1]
                else:
                    print("Intersection point is the last point on the current long path")
            
            # Find on which side the shared intersection point is. Point may not be exactly on the branch, use a 5x5 kernel
            base_start_point = self.long_path_pixel_coords[0]
            base_end_point = self.long_path_pixel_coords[-1]
            new_branch_start_point = self.filament.branch_pts(True)[new_branch_index][0]
            new_branch_end_point = self.filament.branch_pts(True)[new_branch_index][-1]
            
            base_intersection_near_start = self.PointInNeighborhood(intersection, base_start_point)
            new_branch_intersection_near_start = self.PointInNeighborhood(intersection, new_branch_start_point)
            
            print("start_base: {0}; end_base: {1}; start_branch: {2}; end_branch: {3}".format(base_start_point, base_end_point, new_branch_start_point, new_branch_end_point))

            # Add to the longpath coords, subject to the location of the shared intersection point
            if base_intersection_near_start and new_branch_intersection_near_start:
                # Start-start connection, reverse new branch order and insert at front of list
                print("start-start")
                for point in self.filament.branch_pts(True)[new_branch_index]:
                    self.long_path_pixel_coords.insert(0, point)
            elif base_intersection_near_start and not new_branch_intersection_near_start:
                # Start-end connection, insert at front of list
                print("start-end")
                self.long_path_pixel_coords = self.filament.branch_pts(True)[new_branch_index] + self.long_path_pixel_coords
            elif not base_intersection_near_start and new_branch_intersection_near_start:       
                # End-start connection, append to back of list
                print("end-start")
                self.long_path_pixel_coords.extend(self.filament.branch_pts(True)[new_branch_index])
            elif not base_intersection_near_start and not new_branch_intersection_near_start:       
                # End-end connection, reverse new branch order and append to back of list
                print("end-end")
                for i in range(len(self.filament.branch_pts(True)[new_branch_index])-1, -1, -1):
                    self.long_path_pixel_coords.append(self.filament.branch_pts(True)[new_branch_index][i])
            else:
                print("Connection of added branch unclear. Aborting instance.")
                return False
        
        print("New length: {0}".format(len(self.long_path_pixel_coords)))
        return True

        
    def AddThickDotLP(self, origin, size=(7,7)):
        kernel = (np.argwhere(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,size)) + origin - ((np.asarray(size) - 1) / 2).astype('uint8'))
        self.long_path_binary[kernel[:,0], kernel[:,1]] = 1
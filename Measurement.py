import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as u
from skimage import img_as_bool
from skimage.morphology import medial_axis, binary_closing, binary_opening

class Measurement():
    def __init__(self, process_id, raw_frame, binarized_frame):
        self.process_id = process_id
        self.raw_frame = raw_frame
        self.binarized_frame = binarized_frame

        # Image arrays
        self.contour = None
        self.skeleton = None
        self.distance_transform = None

        self.dimensions = np.shape(self.binarized_frame)
        self.long_path_binary = np.zeros(self.dimensions)
        self.tempos = np.zeros(self.dimensions)
        self.processed_frame = np.zeros(self.dimensions)
        self.skeleton_contour = np.zeros(self.dimensions)
        
        # FilFinder data
        self.fil_finder = None
        self.filament = None
        self.fil_length_pixels = 0

        # State information
        self.successful_init = True

        # Longest path assessment variables
        self.longest_path_branch_indices = []
        self.path_endpoints = []
        self.covered_intersec_pt_indices = []
        self.covered_branch_indices = []
        self.number_of_branches = None
        self.standard_length_point = None

        # Apply morphological operations (image processing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(binarized_frame, cv2.MORPH_OPEN, kernel)
        self.contour = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)
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

        # if only one branch, skip the manual exercise
        if self.number_of_branches > 1:
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
                if any(np.equal(self.filament.branch_pts(True)[i], [head_pt[0],head_pt[1]]).all(1)):
                    self.longest_path_branch_indices.append(i)
                    self.covered_branch_indices.append(i)
                    base_branch_index = i
                    print("starting branch is branch {0}".format(i))
                    
                    for y, x in self.filament.branch_pts(True)[i]:
                        self.tempos[y][x] = 1
                    
                    image = cv2.resize(np.rint((self.tempos * 255 + self.contour)/2).astype('uint8'), None, fy=0.7, fx=0.7)
                    cv2.imshow("Evolution", image)
                    cv2.waitKey(0)

                    break
                else:
                    print("starting branch is NOT branch {0}".format(i))

            self.RecursiveBranching(base_branch_index)

            # Fill in the image matrices & grab combined length
            for branch_index in self.longest_path_branch_indices:
                self.fil_length_pixels += self.filament.branch_properties["length"][branch_index].value
                
                for y, x in self.filament.branch_pts(True)[branch_index]:
                    self.long_path_binary[y][x] = 1
             
            for y, x in self.path_endpoints:
                self.long_path_binary[y][x] = 1

        else:
            self.fil_length_pixels = self.filament.length(u.pix).value

            # Fill in the image matrices
            long_path_coords = list(zip(self.filament.longpath_pixel_coords[0], self.filament.longpath_pixel_coords[1]))
            for y, x in long_path_coords:
                self.long_path_binary[y][x] = 1

            # Get the end points of the filament
            # There should only be two since it's a single path
            self.path_endpoints = self.filament.end_pts
            for y, x in self.path_endpoints:
                self.long_path_binary[y][x] = 1

        # Both the end points and self.long_path_binary are now assembled, code pathing logic merges here
        print("Truncating tail...")
        self.TruncateTail()

        print("Adding contour distances...")
        if not self.AddBoundaryDistances():
            return False
        
        self.skeleton_contour = np.rint((self.long_path_binary * 255 + self.contour)/2).astype('uint8')
        self.processed_frame = cv2.addWeighted(self.skeleton_contour,0.65,cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY).astype('uint8'),0.35,0)
        
        image = self.processed_frame
        temp = cv2.resize(image, None, fy=0.8, fx=0.8)
        cv2.imshow("processed", temp)
        cv2.waitKey(0)
        
        return True

    def RecursiveBranching(self, base_branch_index):
        if len(self.covered_intersec_pt_indices) == len(self.filament.intersec_pts):
            # There are no more intersection points to evaluate, we've looked at them all
            print("We're done-so's, all intersection points have been assessed")

            found_point = False
            for end_pt in self.filament.end_pts:
                if any(np.equal(self.filament.branch_pts(True)[base_branch_index], [end_pt[0],end_pt[1]]).all(1)):
                    print("end point {0} found in branch {1}".format(end_pt, base_branch_index))
                    self.path_endpoints.append(end_pt)
                    found_point = True

            if not found_point:
                print("Did not find the endpoint of the terminal branch")

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
                    kernel = [(point[0]+x, point[1]+y) for y in range(-2, 3) for x in range(-2, 3)]
                    print("Kernel: {0}".format(kernel))

                    if any(x in kernel for x in list(map(tuple, self.filament.branch_pts(True)[base_branch_index]))):
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
                for end_pt in self.filament.end_pts:
                    if any(np.equal(self.filament.branch_pts(True)[base_branch_index], [end_pt[0],end_pt[1]]).all(1)):
                        print("end point {0} found in branch {1}".format(end_pt, base_branch_index))
                        self.path_endpoints.append(end_pt)
                        found_point = True

                if not found_point:
                    print("Did not find the endpoint of the terminal branch")
            else:
                # Find which other branches also share this intersection point(s)
                connected_branch_indices = []
                for contending_branch_index in [i for i in range(self.number_of_branches) if i not in self.covered_branch_indices]:
                    for index in intersec_pts_on_basebranch_indices:
                        contending_intersec_pts = self.filament.intersec_pts[index] if isinstance(self.filament.intersec_pts[index], list) else [self.filament.intersec_pts[index]]
                        for point in contending_intersec_pts:
                            # Again, use a kernel to evaluate the neighborhood
                            kernel = [(point[0]+x, point[1]+y) for y in range(-2, 3) for x in range(-2, 3)]

                            if any(x in kernel for x in list(map(tuple, self.filament.branch_pts(True)[contending_branch_index]))):
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
                print("Branch {0} is most aligned, adding".format(best_aligned_index))
                self.longest_path_branch_indices.append(best_aligned_index)
                
                for y, x in self.filament.branch_pts(True)[best_aligned_index]:
                    self.tempos[y][x] = 1
                
                image = cv2.resize(np.rint((self.tempos * 255 + self.contour)/2).astype('uint8'), None, fy=0.7, fx=0.7)
                cv2.imshow("Evolution", image)
                cv2.waitKey(0)
                
                self.RecursiveBranching(best_aligned_index)

    def TruncateTail(self):
        # Find the point along the longest path at which the distance to the contour is minimal
        longpath_mask_array = np.invert(np.array(self.long_path_binary, dtype=bool))
        longpath_masked_array = np.ma.masked_array(self.distance_transform, mask=longpath_mask_array)
        unzipped_min_coords = np.where(longpath_masked_array == np.ma.masked_array.min(longpath_masked_array))
        self.standard_length_point = (unzipped_min_coords[0][0], unzipped_min_coords[1][0])
        print("Standard length point found at {0}".format(self.standard_length_point))

        # Clear the longpath between this point and the end point at the tail. Operate in the box defined between the two points
        tail_sub_array = self.long_path_binary[self.standard_length_point[0]:self.path_endpoints[1][0], self.standard_length_point[1]:self.path_endpoints[1][1]]
        self.fil_length_pixels -= np.sum(tail_sub_array)
        print("Removing {0} pixels from long path".format(np.sum(tail_sub_array)))

        self.long_path_binary[self.standard_length_point[0]:self.path_endpoints[1][0], self.standard_length_point[1]:self.path_endpoints[1][1]] = 0

    def AddBoundaryDistances(self):
        # Get the points that are "pulled back" from the origin point, from which the added distance lines will stem
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
            image = cv2.resize(np.rint((circle_mask + self.long_path_binary)/2 * 255).astype('uint8'), None, fy=0.7, fx=0.7)
            cv2.imshow("Circle Mask", image)
            cv2.waitKey(0)
            return False
            
        pullback_pts.append(head_pullback_pt)
        print("Head pullback point at {0}".format(head_pullback_pt))

        # For the tail point, use the standard length point if it's not equal to the end point
        # otherwise, also use a circle mask
        print("SLP: {0}; Tailpoint: {1}; Equal?: {2}".format(self.standard_length_point, self.path_endpoints[1], self.standard_length_point == self.path_endpoints[1]))
        if (self.standard_length_point != self.path_endpoints[1]):
            pullback_pts.append(self.standard_length_point)
            print("Using standard length point for tail pullback, at {0}".format(self.standard_length_point))
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
            pullback_pt = list(zip(*np.where(combined_array > 1.5)))[0]
            pullback_pts.append(pullback_pt)
            print("Standard length point == end point using circle radius point for tail pullback instead, at {0}".format(pullback_pt))

            # We have to remove the added length if this case occurs
            #### Need to consider orientations for indexing, AND need to remove partial branches directly due to non-lineartity
            self.fil_length_pixels -= np.sum(self.long_path_binary[pullback_pt[0]:self.path_endpoints[1][0], pullback_pt[1]:self.path_endpoints[1][1]])
            self.long_path_binary[pullback_pt[0]:self.path_endpoints[1][0], pullback_pt[1]:self.path_endpoints[1][1]] = 0














        # Run the assessment for each origin/end point pair
        for i in range(2):
            # First create the line mask through the origin/end points
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
            # There will be multiple points since the contour is not one-pixel thick and an infinite line
            # originating in an enclosed shape will necessarily cross it twice
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
                print("Closest boundary point to HEAD {0}, is {1}".format(end_pt, closest_boundary_point))
                
                # Get the slicing direction right, because numpy's an idiot
                self.long_path_binary[end_pt[0]:closest_boundary_point[0]:1 if end_pt[0]<=closest_boundary_point[0] else -1,\
                                      end_pt[1]:closest_boundary_point[1]:1 if end_pt[1]<=closest_boundary_point[1] else -1] = \
                line_mask[end_pt[0]:closest_boundary_point[0]:1 if end_pt[0]<=closest_boundary_point[0] else -1,\
                          end_pt[1]:closest_boundary_point[1]:1 if end_pt[1]<=closest_boundary_point[1] else -1]
                
                distance = np.linalg.norm(np.asarray(end_pt) - np.asarray(closest_boundary_point))
                print("Added head line to path, measuring {0:.2f} pixels".format(distance))
            else:
                print("Closest boundary point to TAIL {0}, is {1}".format(pullback_pt, closest_boundary_point))

                # Assessing the tail
                self.long_path_binary[pullback_pt[0]:closest_boundary_point[0]:1 if pullback_pt[0]<=closest_boundary_point[0] else -1,\
                                      pullback_pt[1]:closest_boundary_point[1]:1 if pullback_pt[1]<=closest_boundary_point[1] else -1] = \
                line_mask[pullback_pt[0]:closest_boundary_point[0]:1 if pullback_pt[0]<=closest_boundary_point[0] else -1,\
                          pullback_pt[1]:closest_boundary_point[1]:1 if pullback_pt[1]<=closest_boundary_point[1] else -1]
                
                distance = np.linalg.norm(np.asarray(pullback_pt) - np.asarray(closest_boundary_point))
                print("Added tail line to path, measuring {0:.2f} pixels".format(distance))

            self.fil_length_pixels += distance
        
        return True
        


from Cameras import Cameras
import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as u
import statistics
import os
from datetime import datetime
from skimage import img_as_bool
from skimage.morphology import medial_axis, binary_closing, binary_opening
import pandas as pd
import matplotlib.pyplot as plt

class Measurement():
    def __init__(self, process_id, raw_frame, binarized_frame):
        self.process_id = process_id
        self.raw_frame = raw_frame
        self.binarized_frame = binarized_frame
        
        # Image arrays
        self.contour = None
        self.skeleton = None
        self.distance_transform = None
        
        self.processed_frame = raw_frame
        self.long_path = None
        
        # FilFinder data
        self.fil_finder = None
        self.filament = None
        self.fil_length_cm = 0
        self.fil_length_pixels = 0
        
        # State information
        self.successful_init = True
        
        # Apply morphological operations (image processing)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(binarized_frame, cv2.MORPH_OPEN, kernel)
        self.contour = cv2.addWeighted(opening,0.5,cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel) ,0.5,0)
        
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
    
    def ConstructLongestPathMask(self):
        self.number_of_branches = len(self.filament.branch_properties["length"])
        print("{0} branches".format(self.number_of_branches))
        
        # if only one branch, skip all of this
        if self.number_of_branches > 1:
            
            # Initialize longest path assessment variables
            self.longest_path_branches = []
            self.covered_intersec_pt_indices = []
            self.filament.rht_branch_analysis()
            
            # Find the starting point: the skeleton end point in the head
            head_pt = None
            for i in range(len(self.filament.end_pts)):
                contending_pt = None
                
                # The entry may be a list of closely located points
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
                        
            print(self.distance_transform[head_pt])
            print(head_pt)
            
            # Fetch the branch that contains this head point
            base_branch_index = None
            for i in range(self.number_of_branches):
                if head_pt in self.filament.branch_pts(True)[i]:
                    self.longest_path_branches.append(i)
                    base_branch_index = i
                    print("starting branch is branch {0}".format(i))
                    
                    break
                else:
                    print("starting branch is NOT branch {0}".format(i))
            
            self.RecursiveBranching(base_branch_index)
            
            # construct the image thing
            # longest path branches list already asasembled
            # AssessFilament()
            
            # Have the AssessFilament() method take the same argument regardless as to the case
        else:
            print("use long_path functionality")
            # AssessFilament()
            
            
            
            
            
        # list of lists
        print("{0} length(s); {1}".format(len(self.filament.branch_properties["length"]), self.filament.branch_properties["length"]))
        print("{0} pixel(s); {1}".format(len(self.filament.branch_pts(True)), self.filament.branch_pts(True)))
        # can be more than one point per
        print("{0} end_pt(s); {1}".format(len(self.filament.end_pts), self.filament.end_pts))
        print("{0} intersec(s); {1}".format(len(self.filament.intersec_pts), self.filament.intersec_pts))
        print("{0} orientation(s); {1}".format(len(self.filament.orientation_branches), self.filament.orientation_branches))
        
        image = cv2.resize(self.skeleton, None, fy=0.7, fx=0.7)
        cv2.imshow("skeleton", image)
        cv2.waitKey(0)
        
    def RecursiveBranching(self, base_branch_index):
        if len(self.covered_intersec_pt_indices) == len(self.filament.intersec_pts):
            # There are no more intersection points to evaluate, we've looked at them all
            print("we're done-so's")
        else:
            # Get the intersection point(s) on the base branch that haven't been covered
            next_pt_indices = []
            contending_pt_indices = [index for index in range(len(self.filament.intersec_pts)) if index not in self.covered_intersec_pt_indices]
            for i in contending_pt_indices:
                contending_pts = self.filament.intersec_pts[i] if isinstance(self.filament.intersec_pts[i], list) else [self.filament.intersec_pts[i]]
                for point in contending_pts:
                    if point in self.filament.branch_pts(True)[base_branch_index]:
                        next_pt_indices.append(i)
                        if i not in self.covered_intersec_pt_indices:
                            self.covered_intersec_pt_indices.append(i)
                        print("Intersection point exists on base branch at index {0}".format(i))
                    else:
                        print("Intersection point at index {0} doesn't exist on base branch".format(i))
            
            if not next_pt_indices:
                # The end of the branch must be an endpoint, we're done
                print("we're also done-so's")
            else:
                # Find which other branches also share this intersection point(s)
                contending_branch_indices = []
                for contending_branch_index in [i for i in range(self.number_of_branches) if i not in self.longest_path_branches]:
                    for index in next_pt_indices:
                        contending_pts = self.filament.intersec_pts[index] if isinstance(self.filament.intersec_pts[index], list) else [self.filament.intersec_pts[index]]
                        for point in contending_pts:
                            if point in self.filament.branch_pts(True)[contending_branch_index]:
                                contending_branch_indices.append(contending_branch_index)
                            
                # Compare the orientations of all contending branches to the base branch to select the next leg
                base_branch_orientation = self.filament.orientation_branches[base_branch_index]
                
                best_aligned_index = None
                best_aligned_distance = None
                for branch_index in contending_branch_indices:
                    contending_branch_orientation = self.filament.orientation_branches[branch_index]
                    contending_distance = abs(base_branch_orientation - contending_branch_orientation)
                    
                    if best_aligned_index is None:
                        best_aligned_index = branch_index
                        best_aligned_distance = contending_distance
                    else:
                        if contending_distance < best_aligned_distance:
                            print("branch {0}: {1} is better aligned than branch {2}: {3}".format(branch_index, contending_distance, best_aligned_index, best_aligned_distance))
                            best_aligned_index = branch_index
                            best_aligned_distance = contending_distance
                    
                # Add this branch to the longest path and run through again
                self.longest_path_branches.append(best_aligned_index)
                self.RecursiveBranching(best_aligned_index)
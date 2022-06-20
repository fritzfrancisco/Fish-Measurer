from Cameras import Cameras
import cv2
import numpy as np
import numpy.ma as ma
from fil_finder import FilFinder2D
import astropy.units as u
import math
import os
from datetime import datetime
from skimage.morphology import skeletonize, thin, medial_axis

class MeasurerInstance():
    def __init__(self, outputFolder, format, min_skel_size=0.01, max_curvature=50):
        ## 26.35
        self.outputFolder = outputFolder
        self.format = format
        self.background = None
        MeasurerInstance.threshold = None 
        MeasurerInstance.fishID = None
        MeasurerInstance.addText = None

        self.max_curvature = math.pi / 180 * max_curvature
        self.min_skel_size = MeasurerInstance.ConvertPixelsToLength(min_skel_size)
        
        Cameras.ConnectMeasurer(self)
        
    def ConvertPixelsToLength(pixels):
        return pixels * 5000

    def ProcessImage(self, frame):
        fgmask = self.fgbg.apply(frame, learningRate=0)
        self.im_bw = cv2.threshold(fgmask, MeasurerInstance.threshold, 255, cv2.THRESH_BINARY)[1]
        
        return self.im_bw
        
    def TrainBackground(self):
        (background_images, empty) = Cameras.GetFixedNumFrames(Cameras.framerate * 3)
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        for image in background_images:
            fgmask = self.fgbg.apply(image)
            
        self.background = fgmask
    
    def SkeletonizeFrames(self, frames):
        self.current_best = {}
        self.current_images = {}
        
        (raw, binarized) = frames
        
        for i in range(len(raw)):
            start_time = datetime.now()
            # Apply morphological operations (image processing)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closing = cv2.morphologyEx(binarized[i], cv2.MORPH_CLOSE, kernel, iterations=3)
            opening = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel) 
            self.processed_image = cv2.addWeighted(cv2.cvtColor(raw[i], cv2.COLOR_BGR2GRAY),0.7,opening,0.3,0)
            self.processed_image = cv2.addWeighted(self.gradient,0.2,self.processed_image,0.8,0)
            
            print("morph ops: " + str((datetime.now() - start_time).total_seconds()))
            start_time = datetime.now()
            
            skeleton_mask = cv2.ximgproc.thinning(opening)
            print("skeletonization cv2: " + str((datetime.now() - start_time).total_seconds()))
            binarized_open = np.round_(opening / 255).astype('uint8')
            start_time = datetime.now()
            
            # skeleton = skeletonize(binarized_open)
            
            # print("skeletonization scikit: " + str((datetime.now() - start_time).total_seconds()))
            # start_time = datetime.now()
            
            # # thinned = thin(binarized_open)
            
            # # print("thinning scikit: " + str((datetime.now() - start_time).total_seconds()))
            # # start_time = datetime.now()
            
            # skel, distance = medial_axis(binarized_open, return_distance=True)
            
            # print("skel med axis scikit: " + str((datetime.now() - start_time).total_seconds()))
            # start_time = datetime.now()
            
            # skel_list = [skeleton_mask, opening, np.round_(skeleton * 255).astype('uint8'), np.round_(skel * 255).astype('uint8')]
            
            # for item in skel_list:
            #     temp = cv2.resize(item, None, fy=0.3, fx=0.3)
            #     cv2.imshow("skele", temp) 
            #     cv2.waitKey(0)
            
            fil = FilFinder2D(skeleton_mask, distance=1500*u.pix, mask=skeleton_mask)
            fil.create_mask(verbose=False, use_existing_mask=True)
            fil.medskel(verbose=False)
            
            # Skeletons must be at least 50 pixels long to count
            fil.analyze_skeletons(skel_thresh=self.min_skel_size*u.pix)
            
            print("filament creation: " + str((datetime.now() - start_time).total_seconds()))
            start_time = datetime.now()
            
            # Attempt to grab the relevant filament
            filament = None
            try:
                if len(fil.lengths()) > 1:
                    lengths = [q.value for q in fil.lengths()]
                    index = lengths.index(max(fil.lengths()).value)
                    filament = fil.filaments[index]
                else:
                    filament = fil.filaments[0]
            except:
                print("could not grab filament")
                continue
                
            long_path = fil.skeleton_longpath
            self.current_images = {"processed": self.processed_image, "contour": self.gradient, "threshed": opening, "raw": cv2.cvtColor(raw[i], cv2.COLOR_BGR2GRAY), "long_path": long_path}
            
            (accepted, statement) = self.AssessFilament(filament)
            print("assess filament: " + str((datetime.now() - start_time).total_seconds()))
            start_time = datetime.now()
            
            print("frame: " + str(i) + "; " + statement)
            if not accepted:
                continue
            else:
                self.current_best["frame"] = i

        if not self.current_best:
            print("dict empty")
        else:
            print("\nFinal: " + str(self.current_best["curvature"]) + "; " + str(self.current_best["length"]) + "; " + str(self.current_best["frame"]))
            
            chosen_image = MeasurerInstance.WatermarkImage(self.current_best)
            
            # Save it and open it
            state = cv2.imwrite(os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + str(self.format)), chosen_image)
            print(state)
    
    def WatermarkImage(current_best):
        # Watermark the results
        chosen_image = cv2.putText(current_best["images"]["processed"], 
                                   "Curvature (deg): " + "{:.2f}".format(current_best["curvature"] * 180 / math.pi) + "; Length (m): " + \
                                       "{:.2f}".format(current_best["length"]),
                                   (15, current_best["images"]["processed"].shape[0]-30), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        # Add metadata
        chosen_image = cv2.putText(chosen_image, datetime.now().strftime("%d.%m.%Y %H:%M:%S"), (15, 70), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
            chosen_image = cv2.putText(chosen_image, "Fish ID: " + MeasurerInstance.fishID, (15, 160), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)

        if MeasurerInstance.addText != None and MeasurerInstance.addText != '':
            text = MeasurerInstance.addText
            y0, dy = 250, 75
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                chosen_image = cv2.putText(chosen_image, line, (15, y), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        return chosen_image
    
    def AssessFilament(self, filament):
        
        fil_length = filament.length(u.pix).value
        added_dist = 0
        
        filament.rht_analysis()
        fil_curvature = filament.curvature.value
        fil_orientation = filament.orientation.value
        
        longpath_pixel_coords = filament.longpath_pixel_coords
        end_pts = [(longpath_pixel_coords[0][0], longpath_pixel_coords[1][0]),
                   (longpath_pixel_coords[0][-1], longpath_pixel_coords[1][-1])]
        
        dimensions = np.shape(self.current_images["processed"])
        slope = math.cos(fil_orientation) / math.sin(fil_orientation)
        
        # run the intersection exercise for each end point
        for point in end_pts:
            # Get the line equation passing through the end point
            line_mask = np.zeros(dimensions)
            added_lines = np.zeros(dimensions)
            
            b = dimensions[0] - point[0] - slope * (point[1])
            
            for x in range(dimensions[1]):
                y_prev = round(slope * (x - 1) + b)
                y = round(slope * x + b)
                y_next = round(slope * (x + 1) + b)
                
                y_start = y_prev + round((y - y_prev) / 2)
                y_end = y + round((y_next - y) / 2)
                
                y_step = 1 if y_start <= y_end else -1
                for y in range(y_start, y_end, y_step):
                    if y < dimensions[0] - 1 and y >= 0:
                        line_mask[dimensions[0] - 1 - y][x] = 1
            
            # Find where the fish boundary and the line intersect
            # There will be multiple points since the contour is not one-pixel thick
            contour_array_normalized = np.round(self.current_images["contour"] / 255)
            combined_array = np.add(line_mask, contour_array_normalized)
            pts_of_interest = zip(*np.where(combined_array > 1.5)) 
            
            reference = np.array(point)
            
            # Get minimum distance end point to contour and add to the filament length
            minimum_distance = None
            min_point_set = None
            for y, x in pts_of_interest:
                coord = np.array([y, x])
                dist = np.linalg.norm(coord - reference) # always >= 0
                if minimum_distance is None:
                    minimum_distance = dist
                    min_point_set = coord
                elif minimum_distance > dist:
                    minimum_distance = dist
                    min_point_set = coord
                        
            fil_length += minimum_distance
            added_dist += minimum_distance
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            added_lines[min_point_set[0]][min_point_set[1]] = 1
            added_lines = cv2.dilate(added_lines, kernel, iterations=3)
            
            # Fill in the skeletonized long path array with these extensions
            step_size = 1 if point[1] <= min_point_set[1] else -1
            for x in range(point[1], min_point_set[1], step_size):
                y_prev = round(slope * (x - step_size) + b)
                y = round(slope * x + b)
                y_next = round(slope * (x + step_size) + b)
                
                # Fill in as many pixels as needed to keep a continuous line,
                # keeping to a 50% tributary width (hence dviding by 2)
                y_start = y_prev + round((y - y_prev) / 2)
                y_end = y + round((y_next - y) / 2)
                
                y_step = 1 if y_start <= y_end else -1
                for y in range(y_start, y_end, y_step):
                    if y < dimensions[0] - 1 and y >= 0:
                        added_lines[dimensions[0] - 1 - y][x] = 1
                    
            added_lines = cv2.dilate(added_lines, kernel)
            self.current_images["long_path"] = np.add(self.current_images["long_path"], added_lines)

        self.current_images["long_path"] = np.where(self.current_images["long_path"] > 1, 1, self.current_images["long_path"])
        temp = cv2.resize(self.current_images["long_path"], None, fy=0.5, fx=0.5)
        cv2.imshow("long_path", temp) 
        cv2.waitKey(0)
        
        print(slope, fil_curvature, "fil length: " + str(fil_length), "added length: " + str(added_dist))
            
        # Compare to current best filament
        if not any(self.current_best):
            return self.CompareFilamentProperties(fil_curvature, fil_length) # will create the dictionary
        elif fil_length <= self.current_best["length"]:
            return (False, "filament too short")
        else:
            return self.CompareFilamentProperties(fil_curvature, fil_length)

    def ShowImage(image1, image2, resize=0.7, name='Image', pausekey=False, show=False):
        image = cv2.addWeighted(image1,0.5,image2,0.5,0)
        if show:
            temp = cv2.resize(image, None, fy=resize, fx=resize)
            cv2.imshow(name, temp) 
            if pausekey:
                cv2.waitKey(0)
            
        return image
    
    def CompareFilamentProperties(self, fil_curvature, fil_length):
        if fil_curvature < self.max_curvature:
            image = MeasurerInstance.ShowImage(self.current_images["processed"], (self.current_images["long_path"] * 255).astype('uint8'), name="binarized") 
            self.current_images["processed"] = image
            self.current_best = {"curvature": fil_curvature, "length": fil_length, "images": self.current_images}
            return (True, "candidate accepted")
        else:
            return (False, "candidate too curved")
        
        
        
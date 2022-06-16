from Cameras import Cameras
import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as u
import math
import os
from datetime import datetime

class MeasurerInstance():
    def __init__(self, outputFolder, format, min_skel_size=0.01, max_curvature=50):
        ## 26.35
        self.outputFolder = outputFolder
        self.format = format
        self.background = MeasurerInstance.threshold = MeasurerInstance.fishID = MeasurerInstance.addText = None

        self.max_curvature = math.pi / 180 * max_curvature
        self.min_skel_size = MeasurerInstance.ConvertPixelsToLength(min_skel_size)
        
        Cameras.ConnectMeasurer(self)
        
    def ConvertPixelsToLength(pixels):
        return pixels * 5000

    def ProcessImage(self, frame):
        self.frame = frame
        fgmask = self.fgbg.apply(frame, learningRate=0)
        self.im_bw = cv2.threshold(fgmask, MeasurerInstance.threshold, 255, cv2.THRESH_BINARY)[1]
        
        return self.im_bw
        
    def TrainBackground(self):
        background_images = Cameras.GetFixedNumFrames(Cameras.framerate * 3)
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        for image in background_images:
            fgmask = self.fgbg.apply(image)
            
        self.background = fgmask
    
    def SkeletonizeFrames(self, frames):
        self.current_best = {}
        self.current_images = {}
        
        for i, frame in enumerate(frames):            
            # Apply morphological operations (image processing)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=3)
            opening = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel) 
            self.processed_image = cv2.addWeighted(cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY),0.7,opening,0.3,0)
            self.processed_image = cv2.addWeighted(self.gradient,0.2,self.processed_image,0.8,0)
            
            skeleton_mask = cv2.ximgproc.thinning(opening)
            fil = FilFinder2D(skeleton_mask, distance=1500*u.pix, mask=skeleton_mask)
            fil.create_mask(verbose=False, use_existing_mask=True)
            fil.medskel(verbose=False)
            
            # Skeletons must be at least 50 pixels long to count
            fil.analyze_skeletons(skel_thresh=self.min_skel_size*u.pix)
            
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
            self.current_images = {"processed": self.processed_image, "contour": self.gradient, "threshed": opening, "raw": self.frame, "long_path": long_path}
            
            (accepted, statement) = self.AssessFilament(filament)
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
            
            cv2.imshow("Final Image", chosen_image) 
            cv2.waitKey(0)
    
    def WatermarkImage(current_best):
        # Watermark the results
        chosen_image = cv2.putText(current_best["images"]["processed"], 
                                   "curvature (deg): " + str(current_best["curvature"] * 180 / math.pi) + "; length (m): " + \
                                       str(MeasurerInstance.ConvertPixelsToLength(current_best["length"])),
                                   (15, current_best["images"]["processed"].shape[0]-20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)
        
        # Add metadata
        chosen_image = cv2.putText(chosen_image, datetime.now().strftime("%d.%m.%Y %H:%M:%S"), (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)
        
        if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
            chosen_image = cv2.putText(chosen_image, "Fish ID: " + MeasurerInstance.fishID, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)

        if MeasurerInstance.addText != None and MeasurerInstance.addText != '':
            text = MeasurerInstance.addText
            y0, dy = 95, 25
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                chosen_image = cv2.putText(chosen_image, line, (15, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)
        
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
            b = dimensions[0] - point[0] - slope * (point[1])
            
            for x in range(dimensions[1]):
                y = round(slope * x + b)
                if y < dimensions[0] - 1 and y >= 0:
                    line_mask[dimensions[0] - 1 - y, x] = 1
                    
            # Find where the fish boundary and the line intersect
            # There will be multiple points since the contour is not one-pixel thick
            combined_array = line_mask + self.current_images["contour"]
            pts_of_interest = np.where(combined_array > 1)
            
            # Get minimum distance end point to contour and add to the filament length
            minimum_distance = None
            min_point_set = None
            for i in range(len(pts_of_interest[0])):
                coord = np.array([pts_of_interest[0][i], pts_of_interest[1][i]])
                dist = np.linalg.norm(coord - np.array(point))
                if dist < 0: # endpoint on edge of a multi-pixel thick contour
                    minimum_distance = dist
                    min_point_set = coord
                    break
                else:
                    if minimum_distance is None:
                        minimum_distance = dist
                        min_point_set = coord
                    elif minimum_distance > dist:
                        minimum_distance = dist
                        min_point_set = coord
                        
            fil_length += minimum_distance
            added_dist += minimum_distance
            
            # Fill in the skeletonized long path array with these extensions
            for x in range(point[1], min_point_set[1], 1 if point[1] <= min_point_set[1] else -1):
                y = round(slope * x + b)
                if y < dimensions[0] - 1 and y >= 0:
                    self.current_images["long_path"][dimensions[0] - 1 - y][x] = 1
                
        print(slope, fil_curvature, "fil length: " + str(fil_length), "added length: " + str(added_dist))
            
        # Compare to current best filament
        if not any(self.current_best):
            return self.CompareFilamentProperties(fil_curvature, fil_length) # will create the dictionary
        elif fil_length <= self.current_best["length"]:
            return (False, "filament too short")
        else:
            return self.CompareFilamentProperties(fil_curvature, fil_length)


    def ShowImage(image1, image2, resize=0.7, name='Image', pausekey=False):
        image = cv2.addWeighted(image1,0.5,image2,0.5,0)
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
        
        
        
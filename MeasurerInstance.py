from Cameras import Cameras
import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as u
import math
import statistics
import os
from datetime import datetime
from skimage.morphology import skeletonize
import pandas as pd

class MeasurerInstance():
    def __init__(self, outputFolder, format, min_skel_size=0.01, max_curvature=50):
        ## 26.35
        self.outputFolder = outputFolder
        self.format = format
        self.background = None
        self.filament_lengths = []
        self.length_avg = None
        self.stdev = None
        MeasurerInstance.threshold = None 
        MeasurerInstance.fishID = None
        MeasurerInstance.addText = None
        MeasurerInstance.processingFrame = None

        self.max_curvature = math.pi / 180 * max_curvature
        self.min_skel_size = MeasurerInstance.ConvertLengthToPixels(min_skel_size)
        
        Cameras.ConnectMeasurer(self)
        
    def ConvertPixelsToLength(pixels):
        return (pixels + 12.169) / 10.783

    def ConvertLengthToPixels(length):
        return 10.783 * length - 12.169

    def ProcessImage(self, frame):
        fgmask = self.fgbg.apply(frame, learningRate=0)
        fully_binarized = cv2.threshold(fgmask, MeasurerInstance.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Fill all contours and only show contour with largest area
        contour, hier = cv2.findContours(fully_binarized,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        biggestContour = max(contour, key = cv2.contourArea)
        
        self.im_bw = np.zeros(np.shape(fully_binarized)).astype('uint8')
        cv2.drawContours(self.im_bw, [biggestContour],-1,255,thickness=cv2.FILLED)
        
        return self.im_bw
        
    def TrainBackground(self):
        (background_images, empty) = Cameras.GetFixedNumFrames(Cameras.framerate * 3)
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        for image in background_images:
            fgmask = self.fgbg.apply(image)
            
        self.background = fgmask
    
    def SkeletonizeFrames(self, frames):
        self.measurements = {}
        self.current_images = {}
        
        (raw, binarized) = frames
        frames_path = os.path.join(self.outputFolder, "frames")
        if not os.path.isdir(frames_path):
            os.mkdir(frames_path)
        
        for i in range(len(raw)):
            MeasurerInstance.processingFrame = i
            
            # Apply morphological operations (image processing)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            opening = cv2.morphologyEx(binarized[i], cv2.MORPH_OPEN, kernel)
            self.gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel) 
            
            self.gradientOpen = cv2.addWeighted(opening,0.5,self.gradient,0.5,0)
            self.processed_image = raw[i]
            
            # skeleton = cv2.ximgproc.thinning(opening)
            skeleton = skeletonize(np.round_((opening / 255).astype('uint8')))
            
            fil = FilFinder2D(skeleton, distance=1500*u.pix, mask=skeleton)
            fil.create_mask(verbose=False, use_existing_mask=True)
            fil.medskel(verbose=False)
            
            # Skeletons must be at least 50 pixels long to count
            try:
                fil.analyze_skeletons(skel_thresh=self.min_skel_size*u.pix)
            except ValueError:
                print("Filfinder error")
                continue
            
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
                
            self.current_images = {"processed": self.processed_image, "contour": self.gradient, "threshed": opening, "raw": cv2.cvtColor(raw[i], cv2.COLOR_BGR2GRAY)}
            fil_length, curvature = self.AssessFilament(filament)
            fil_length = MeasurerInstance.ConvertPixelsToLength(fil_length)
            print("frame: " + str(i), "fil length: " + str(fil_length), "curvature: " + str(curvature))
            
            extended_frames_path = os.path.join(frames_path, str(i) + str(self.format))
            state = cv2.imwrite(extended_frames_path, self.current_images["processed"])
            
            # Save the data from the frame
            self.filament_lengths.append((i, fil_length))
            self.measurements[i] = {"length": fil_length, "curvature": curvature, "images": self.current_images}

        MeasurerInstance.processingFrame = None
        
        # Remove outliers
        if self.filament_lengths:
            self.length_avg = statistics.mean([lens for i, lens in self.filament_lengths])
            refined_list = [(fil_length, self.measurements[i]["curvature"], i) for i, fil_length in self.filament_lengths if abs((fil_length - self.length_avg) / self.length_avg) <= 0.1]
            split_list = [list(t) for t in zip(*refined_list)]
            
            self.length_stats = (statistics.mean(split_list[0]), statistics.stdev(split_list[0]))
            self.curve_stats = (statistics.mean(split_list[1]), statistics.stdev(split_list[1])) 
            
            df = pd.DataFrame(data={"frame_number": split_list[2], "length_mm": split_list[0], "curvature_rad": split_list[1]})
            df.to_csv(os.path.join(self.outputFolder, "data_output.csv"), sep=';',index=False) 
            
            # find the instane with the closest length value
            closest_index = split_list[0].index(min(split_list[0], key=lambda x:abs(x-self.length_stats[0])))
            closest_instance = self.measurements[split_list[2][closest_index]]
            closest_length = self.measurements[closest_index]["length"]
            
            print("\nFinal: " + str(self.curve_stats[0]) + "; " + str(self.length_stats[0]) + "; " + str(closest_index))
            chosen_image = MeasurerInstance.WatermarkImage(closest_instance, closest_index, closest_length, self.length_stats, self.curve_stats)
            
            # Save it and open it
            if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
                state = cv2.imwrite(os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + \
                    "_" + str(MeasurerInstance.fishID) + str(self.format)), chosen_image)
            else:
                state = cv2.imwrite(os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + \
                    str(self.format)), chosen_image)
        else:
            print("could not get lengths")
            # could not read anything
        ## IF STATE ERROR MESSAGE
                
    def WatermarkImage(closest_instance, closest_index, closest_length, length_stats, curve_stats):
        # Watermark the results
        chosen_image = cv2.putText(closest_instance["images"]["processed"], 
                                    "Avg. Curvature (deg): " + "{:.2f}".format(curve_stats[0] * 180 / math.pi) + \
                                    " +/- " + "{:.2f}".format(curve_stats[1] * 180 / math.pi),
                                    (15, closest_instance["images"]["processed"].shape[0]-210), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        chosen_image = cv2.putText(chosen_image, 
                                    "Avg. Length: " +  "{:.2f}".format(length_stats[0]) + "mm (" + \
                                    "{:.2f}".format(MeasurerInstance.ConvertLengthToPixels(length_stats[0])) + "pix)" + \
                                    " +/- " + "{:.2f}".format(length_stats[1]) + "mm", (15, chosen_image.shape[0]-120), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        chosen_image = cv2.putText(chosen_image, 
                                    "Frame: " +  "{0}".format(closest_index) + \
                                    "; Length: " + "{:.2f}".format(closest_length) + \
                                    "mm (" + "{:.2f}".format(MeasurerInstance.ConvertLengthToPixels(closest_length)) + "pix)",
                                    (15, chosen_image.shape[0]-30), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
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
        dimensions = np.shape(self.current_images["contour"])
        
        fil_length = filament.length(u.pix).value
        added_dist = 0
        
        filament.rht_analysis()
        fil_curvature = filament.curvature.value
        fil_orientation = filament.orientation.value
        
        # Get the pixels on the longpath and all the [many] branch endpoints
        # the longpath pixels are not in order, and so we need to find the
        # endpoints of this path manually
        longpath_pixel_coords = filament.longpath_pixel_coords
        long_zipped = list(zip(longpath_pixel_coords[0], longpath_pixel_coords[1]))
        end_pt_coords = filament.end_pts
        longpath_pixel_coords_array = np.asarray(longpath_pixel_coords)
        
        end_candidates = []
        list_max = 1000000
        for pt in end_pt_coords:
            # Get the distance from each endpoint to all other points on the longpath
            pt_array = np.asarray(pt).reshape((2,1))
            dist_mat = np.linalg.norm(np.subtract(longpath_pixel_coords_array, pt_array), axis=0)
        
            min_dist = np.amin(dist_mat)
            if len(end_candidates) < 2:
                end_candidates.append((pt, min_dist))
                list_max = max([dist for coord, dist in end_candidates])
            else:
                if min_dist < list_max:
                    pop_index = [dist for coord, dist in end_candidates].index(list_max)
                    end_candidates.pop(pop_index)
                    end_candidates.append((pt, min_dist))
                    list_max = max([dist for coord, dist in end_candidates])
                    
        ## There might be others with 0 distance (ie, at the -1 pixels position). Need to consider
        ## for now leave it, the user can restart
        end_pts = [coord for coord, dist in end_candidates]
        
        # Fill in the image matrices
        longest_path_mat = np.zeros(dimensions)
        for y, x in long_zipped:
            longest_path_mat[y-1][x-1] = 1
        self.current_images["long_path"] = longest_path_mat
        
        # Run the intersection exercise for each end point to determine
        # how much length to add onto the ends of the longpath
        slope = math.cos(fil_orientation) / math.sin(fil_orientation)
        for y_pt, x_pt in end_pts:
            # Get the line equation passing through the end point
            line_mask = np.zeros(dimensions)
            b = dimensions[0] - y_pt - slope * x_pt
            
            for x in range(dimensions[1]):
                y_prev = round(slope * (x - 1) + b)
                y = round(slope * x + b)
                y_next = round(slope * (x + 1) + b)
                
                if y < dimensions[0] - 1 and y >= 0:
                    line_mask[dimensions[0] - 1 - y][x] = 1
                
                # Fill in as many pixels as needed to keep a continuous line,
                # keeping to a 50% tributary width (hence dividing by 2)
                # otherwise we have a dotted line
                if y_prev != y_next:
                    y_start = y_prev + round((y - y_prev) / 2)
                    y_end = y + round((y_next - y) / 2)
                    
                    y_step = 1 if y_start <= y_end else -1
                    for y in range(y_start, y_end, y_step):
                        if y < dimensions[0] - 1 and y >= 0:
                            line_mask[dimensions[0] - 1 - y][x] = 1
            
            # Find where the thresholded fish boundary and the line intersect
            # There will be multiple points since the contour is not one-pixel thick
            contour_array_normalized = np.round(self.current_images["contour"] / 255)
            combined_array = np.add(line_mask, contour_array_normalized)
            
            pts_of_interest = list(zip(*np.where(combined_array > 1.5)))
            reference = np.array([y_pt, x_pt])
            
            # Get minimum distance from end point to contour and add to filament
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
            
            # Fill in the skeletonized long path array with these extensions
            step_size = 1 if x_pt <= min_point_set[1] else -1
            for x in range(x_pt, min_point_set[1] + step_size, step_size):
                y_prev = round(slope * (x - step_size) + b)
                y = round(slope * x + b)
                y_next = round(slope * (x + step_size) + b)
                
                if dimensions[0] - min_point_set[0] >= dimensions[0] - y_pt:
                    if y <= dimensions[0] - min_point_set[0] and y >= dimensions[0] - y_pt:
                        self.current_images["long_path"][dimensions[0] - 1 - y][x] = 1
                else:
                    if y <= dimensions[0] - y_pt and y >= dimensions[0] - min_point_set[0]:
                        self.current_images["long_path"][dimensions[0] - 1 - y][x] = 1
                    
                if y_prev != y_next:
                    y_start = y_prev + round((y - y_prev) / 2)
                    y_end = y + round((y_next - y) / 2)
                    
                    y_step = 1 if y_start <= y_end else -1
                    for y in range(y_start, y_end, y_step):
                        if dimensions[0] - min_point_set[0] >= dimensions[0] - y_pt:
                            if y <= dimensions[0] - min_point_set[0] and y >= dimensions[0] - y_pt:
                                self.current_images["long_path"][dimensions[0] - 1 - y][x] = 1
                        else:
                            if y <= dimensions[0] - y_pt and y >= dimensions[0] - min_point_set[0]:
                                self.current_images["long_path"][dimensions[0] - 1 - y][x] = 1
                    
        self.current_images["long_path"] = np.where(self.current_images["long_path"] > 1, 1, self.current_images["long_path"])        
        
        longOpenGradient = cv2.addWeighted((self.current_images["long_path"] * 255).astype('uint8'),0.6,self.gradientOpen,0.4,0)
        image = MeasurerInstance.ShowImage(self.current_images["processed"], cv2.cvtColor(longOpenGradient, cv2.COLOR_GRAY2RGB), name="binarized") 
        self.current_images["processed"] = image
        
        return fil_length, fil_curvature

    def ShowImage(image1, image2, resize=0.5, name='Image', pausekey=False, show=False):
        image = cv2.addWeighted(image1,0.65,image2,0.35,0)
        if show:
            temp = cv2.resize(image, None, fy=resize, fx=resize)
            cv2.imshow(name, temp) 
            if pausekey:
                cv2.waitKey(0)
            
        return image
        
        
        
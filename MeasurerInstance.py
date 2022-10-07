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

class MeasurerInstance():
    def __init__(self, outputFolder, format):
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
        MeasurerInstance.error = (False, None)
        MeasurerInstance.dist_transform = None

        Cameras.ConnectMeasurer(self)

    def SubtractBackground(self, frame):
        # Subtract the background and binarize via thresholding (according to shadows setting)
        fgmask = self.fgbg.apply(frame, learningRate=0)
        fully_binarized = cv2.threshold(fgmask, MeasurerInstance.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Fill all blob contours and only show the contour with largest area
        try:
            contour, hier = cv2.findContours(fully_binarized,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            biggestContour = max(contour, key = cv2.contourArea)
            
            self.im_bw = np.zeros(np.shape(fully_binarized)).astype('uint8')
            cv2.drawContours(self.im_bw, [biggestContour],-1,255,thickness=cv2.FILLED)
            
            if (MeasurerInstance.error[1] == "The threshold value is set too high and all blobs are being filtered out. Please lower it a bit"):
                MeasurerInstance.error = (False, None)
                
        except ValueError as e:
            print(str(type(e).__name__) + ": The biggestContour list is empty, no blobs are being picked up -->", e)
            self.im_bw = fully_binarized
            MeasurerInstance.error = (True, "The threshold value is set too high and all blobs are being filtered out. Please lower it a bit")
            
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
        self.filament_lengths = []
        
        (raw, binarized) = frames
        
        # Create the destination folder
        target_folder_name = None
        if MeasurerInstance.fishID != None and MeasurerInstance.fishID != '':
            target_folder_name = os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + \
                "_ID-" + str(MeasurerInstance.fishID))
        else:
            target_folder_name = os.path.join(self.outputFolder, str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")))
        
        if not os.path.isdir(target_folder_name):
            os.mkdir(target_folder_name)

        # Create the folder for the individual frames
        frames_path = os.path.join(target_folder_name, "frames")
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
            
            # Skeletonize
            bool_image = img_as_bool(binary_opening(opening))
            
            timer = datetime.now()
            skeleton, MeasurerInstance.dist_transform = medial_axis(bool_image, return_distance=True)
            skeleton = binary_closing(skeleton)
            timing = datetime.now() - timer
            print ("medial took " + str(timing.total_seconds()) + " seconds")
                
            fil = FilFinder2D(skeleton, mask=skeleton)
            fil.create_mask(verbose=False, use_existing_mask=True)
            fil.medskel(verbose=False)
            
            try:
                fil.analyze_skeletons(skel_thresh=1*u.pix)
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
                
            self.current_images = {"processed": self.processed_image, "contour": self.gradient, "threshed": opening, "raw": cv2.cvtColor(raw[i], cv2.COLOR_BGR2GRAY),  "full_skeleton": cv2.addWeighted((skeleton * 255).astype('uint8'), 0.65, self.gradientOpen, 0.35, 0)}
            fil_length_pixels = self.AssessFilament(filament)
            fil_length = Cameras.ConvertPixelsToLength(fil_length_pixels)
            print("frame: " + str(i), "fil length: " + str(fil_length), "pixels: " + str(fil_length_pixels))
            
            extended_frames_path = os.path.join(frames_path, str(i) + str(self.format))
            extended_frames_path2 = os.path.join(frames_path, "full_skeleton_" + str(i) + str(self.format))
            cv2.imwrite(extended_frames_path, self.current_images["processed"])
            cv2.imwrite(extended_frames_path2, self.current_images["full_skeleton"])
            
            # Save the data from the frame
            self.filament_lengths.append((i, fil_length))
            self.measurements[i] = {"length": fil_length, "images": self.current_images}

        MeasurerInstance.processingFrame = None
        
        # Remove outliers
        if self.filament_lengths:
            initial_list = [lens for i, lens in self.filament_lengths]
            self.length_avg = statistics.mean(initial_list)
            print("Avg length: " + str(self.length_avg))
            print("y = {0}x + {1}".format(Cameras.conversion_slope, Cameras.conversion_intercept))
            # print("Total list entries: " + str(initial_list.count()))
            
            refined_list = [(fil_length, self.measurements[i], i) for i, fil_length in self.filament_lengths if abs((fil_length - self.length_avg) / self.length_avg) <= 0.1]
            MeasurerInstance.trial_count = len(refined_list)
            split_list = [list(t) for t in zip(*refined_list)]
            print("Refined list entries: " + str(MeasurerInstance.trial_count))
            
            if not split_list:
                print("All lengths filtered out!")
                MeasurerInstance.error = (True, "The lengths obtained are too variant to be consolidated. The data will not be saved, please re-measure")
            else:
                self.length_stats = (statistics.mean(split_list[0]), statistics.stdev(split_list[0]))
                
                df = pd.DataFrame(data={"frame_number": split_list[2], "length_mm": split_list[0]})
                df.to_csv(os.path.join(target_folder_name, "data-output.csv"), sep=';',index=False) 
                
                # find the instance with the closest length value
                try:
                    local_index = split_list[0].index(min(split_list[0], key=lambda x:abs(x-self.length_stats[0])))
                    
                    closest_instance = split_list[1][local_index]
                    closest_index = split_list[2][local_index]
                except (KeyError):
                    MeasurerInstance.error = (True, "There was an error processing the obtained data. The data wsa not saved, please try again")
                
                print("\nFinal: " + str(self.length_stats[0]) + "; " + str(closest_index))
                chosen_image = MeasurerInstance.WatermarkImage(closest_instance, closest_index, self.length_stats)
                
                # Save it and open it
                state = cv2.imwrite(os.path.join(target_folder_name, "closest-image" + str(self.format)), chosen_image)
        else:
            MeasurerInstance.error = (True, "The length values could not be obtained from the image. Either the blob was too small and filtered out, or the skeletonization process was too complex. Please try again")
                
    def WatermarkImage(closest_instance, closest_index, length_stats):
        # Watermark the results
        chosen_image = cv2.putText(closest_instance["images"]["processed"], 
                                    "Length: " +  "{:.2f}".format(length_stats[0]) + "cm (" + \
                                    "{:.2f}".format(closest_instance["length"]) + "cm)" + \
                                    " +/- " + "{:.2f}".format(length_stats[1]) + "cm", 
                                    (15, closest_instance["images"]["processed"].shape[0]-30), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
        chosen_image = cv2.putText(chosen_image, 
                                    "{0}".format(MeasurerInstance.trial_count) +  " images",
                                    (15, chosen_image.shape[0]-120), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), lineType=cv2.LINE_AA)
        
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
        
        # Get the pixels on the longpath and all the [many] branch endpoints
        # the longpath pixels are not in order, and so we need to find the
        # endpoints of this path manually --> really annoying
        longpath_pixel_coords = filament.longpath_pixel_coords
        longpath_pixel_coords_array = np.asarray(longpath_pixel_coords)
        long_zipped = list(zip(longpath_pixel_coords[0], longpath_pixel_coords[1]))
        
        # Gets the end points of the entire non-pruned filament
        end_pt_coords = filament.end_pts
        
        end_candidates = []
        list_max = 1000000
        for pt in end_pt_coords:
            # Get the distance from each endpoint to all other points on the longpath
            pt_array = np.asarray(pt).reshape((2,1))
            dist_mat = np.linalg.norm(np.subtract(longpath_pixel_coords_array, pt_array), axis=0)

            # The actual endpoints will be very close to (even on top of) at least one
            # of the points on the long path
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
        # the path to add
        for y_pt, x_pt in end_pts:
            distance = MeasurerInstance.dist_transform[y_pt][x_pt]
            fil_length += distance
            added_dist += distance
            print("distance_transform: " + str(distance))
            
            # Start with the appropriate slope for the line mask
            circle_mask = np.zeros(dimensions)
            circle_radius = 20
            for x in range(x_pt - circle_radius, x_pt + circle_radius + 1):
                for y in range(y_pt - circle_radius, y_pt + circle_radius + 1):
                    if x <= dimensions[1] - 1 and x >= 0:
                        if y <= dimensions[0] - 1 and y >= 0:
                            ref_val = ((x-x_pt)**2 + (y-y_pt)**2)**(0.5)
                            if ref_val >= 19.3 and ref_val <= 20.7:
                                circle_mask[y][x] = 1
            
            combined_array = np.add(circle_mask, longest_path_mat)
            pts_of_interest = list(zip(*np.where(combined_array > 1.5)))
            
            # Just take the first point, they should all yield similar results if more than one
            other_point = pts_of_interest[0]
            slope = (y_pt - other_point[0]) / (x_pt -  other_point[1])

            # Get the line equation passing through the end point
            line_mask = np.zeros(dimensions)
            b = y_pt - slope * x_pt
            
            for x in range(dimensions[1]):
                try:
                    y_prev = round(slope * (x - 1) + b)
                    y = round(slope * x + b)
                    y_next = round(slope * (x + 1) + b)
                    
                    if y < dimensions[0] - 1 and y >= 0:
                        line_mask[y][x] = 1
                    
                    # Fill in as many pixels as needed to keep a continuous line,
                    # keeping to a 50% tributary width (hence dividing by 2)
                    # otherwise we have a dotted line
                    if y_prev != y_next:
                        y_start = y_prev + round((y - y_prev) / 2)
                        y_end = y + round((y_next - y) / 2)
                        
                        y_step = 1 if y_start <= y_end else -1
                        for y in range(y_start, y_end, y_step):
                            if y < dimensions[0] - 1 and y >= 0:
                                line_mask[y][x] = 1
                except OverflowError:
                    print("Infinity error")
                    print("x: " + str(x) + "; slope: " + str(slope))
                    continue
            
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
            
            # Fill in the skeletonized long path array with these extensions
            step_size = 1 if x_pt <= min_point_set[1] else -1
            for x in range(x_pt, min_point_set[1] + step_size, step_size):
                y_prev = round(slope * (x - step_size) + b)
                y = round(slope * x + b)
                y_next = round(slope * (x + step_size) + b)
                
                if min_point_set[0] >= y_pt:
                    if y <= min_point_set[0] and y >= y_pt:
                        self.current_images["long_path"][y][x] = 1
                        # self.current_images["long_path"][1 - y][x] = 1
                else:
                    if y <= y_pt and y >= min_point_set[0]:
                        self.current_images["long_path"][y][x] = 1
                    
                if y_prev != y_next:
                    y_start = y_prev + round((y - y_prev) / 2)
                    y_end = y + round((y_next - y) / 2)
                    
                    y_step = 1 if y_start <= y_end else -1
                    for y in range(y_start, y_end, y_step):
                        if min_point_set[0] >= y_pt:
                            if y <= min_point_set[0] and y >= y_pt:
                                self.current_images["long_path"][y][x] = 1
                        else:
                            if y <= y_pt and y >= min_point_set[0]:
                                self.current_images["long_path"][y][x] = 1
                    
        self.current_images["long_path"] = np.where(self.current_images["long_path"] > 1, 1, self.current_images["long_path"])   
        
        longOpenGradient = cv2.addWeighted((self.current_images["long_path"] * 255).astype('uint8'),0.6,self.gradientOpen,0.4,0)
        image = MeasurerInstance.ShowImage(self.current_images["processed"], cv2.cvtColor(longOpenGradient, cv2.COLOR_GRAY2RGB), name="binarized") 
        self.current_images["processed"] = image
        
        return fil_length

    def ShowImage(image1, image2, resize=0.5, name='Image', pausekey=False, show=False):
        image = cv2.addWeighted(image1,0.65,image2,0.35,0)
        if show:
            temp = cv2.resize(image, None, fy=resize, fx=resize)
            cv2.imshow(name, temp) 
            if pausekey:
                cv2.waitKey(0)
            
        return image
        
        
        
from typing import Any, Type
from Cameras import Cameras
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis, skeletonize
import cv2
import numpy as np
from fil_finder import FilFinder2D
import astropy.units as u
import math

class MeasurerInstance():
    
    def __init__(self):
        # self.settings = settingsDict["settings"] 
        # self.watermark = settingsDict["watermark"]
        # self.outputFolder = settingsDict["folder"]
        # self.format = settingsDict["format"]
        
        # Cameras.ApplySettings(exposure=self.settings["exposure"], gain=self.settings["gain"],
        #                       framerate=self.settings["framerate"], duration=self.settings["duration"])
        
        # image_array = Cameras.GetImages()
        
        # return a list of n X m frames with (n,m) being the pixel and the entry the [0,255] color value:
        MeasurerInstance.max_curvature = 0.46
        MeasurerInstance.current_best = {}
        self.image_array = MeasurerInstance.Video_to_Frames()
        
        # masker = self.background.apply(self.image_array[-1])
        # temp = cv2.resize(masker, None, fy=0.3, fx=0.3)
        # cv2.imshow('-1', temp)
        # masker = self.background.apply(self.image_array[110])
        # temp = cv2.resize(masker, None, fy=0.3, fx=0.3)
        # cv2.imshow('110', temp)
        
        # self.ProcessImages()
        
    def Skeletonize(self, img):
        # Compute the medial axis (skeleton) and the distance transform
        skel, distance = medial_axis(img, return_distance=True)

        # Distance to the background for pixels of the skeleton
        dist_on_skel = distance * skel

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[1].imshow(dist_on_skel, cmap='magma')
        
        # filFinder stuff here too
        
        return 1.0 # return length

    def ProcessImages(self):
        allow = True
        
        # for i, img in enumerate(self.image_array):
            
            # get skeleton longest path (fil finder)
            # add distance to contour at each extreme point for full length
            # add length to a list            
            # self.Skeletonize(img)    
        
        # average length across timespan (list)
        # select first image
        # watermark with length (bottom left)
        # save to save location
        # open (?)
    
    def Video_to_Frames():
        images = []

        MeasurerInstance.show = True
        
        small = True
        if small:
            cap = cv2.VideoCapture("C:/Users/james/Downloads/BaslerSmall.mp4")
            while not cap.isOpened():
                cap = cv2.VideoCapture("C:/Users/james/Downloads/BaslerSmall.mp4")
                cv2.waitKey(1000)
        else:
            cap = cv2.VideoCapture("C:/Users/james/Downloads/BaslerBig.mp4")
            while not cap.isOpened():
                cap = cv2.VideoCapture("C:/Users/james/Downloads/BaslerBig.mp4")
                cv2.waitKey(1000)
        
        fgbg = cv2.createBackgroundSubtractorMOG2()
        
        pos_frame = cap.get(1)
        while True:
            frameready, frame = cap.read() # get the frame
            if frameready:
                # Convert to grayscale and save
                fgmask = fgbg.apply(frame)
                # blurred = cv2.GaussianBlur(fgmask, (7, 7), 0)
                im_bw = cv2.threshold(fgmask, 120, 255, cv2.THRESH_BINARY)[1]
                
                # im_bw = cv2.adaptiveThreshold(fgmask, 255,
	            #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                closing = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel, iterations=3)
                opening = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
                
                gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel) 
                
                # get distance from filament end points to each non-zero element of the gradient array
                # - angles are all lines to the left of the y-axis, and + are those laying to the right, all going through the origin
                ## and therefore continuing under the x-axis.
                
                # temp = cv2.resize(gradient, None, fy=0.7, fx=0.7)
                # cv2.imshow('gradient', temp) 
                
                dst = cv2.addWeighted(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),0.7,opening,0.3,0)
                dst = cv2.addWeighted(gradient,0.2,dst,0.8,0)
                
                if cap.get(1) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.65:
                    print("frame: " + str(cap.get(1)))
                    skeleton_mask = cv2.ximgproc.thinning(opening)
                    # https://stackoverflow.com/questions/53481596/python-image-finding-largest-branch-from-image-skeleton
                    fil = FilFinder2D(skeleton_mask, distance=1500*u.pix, mask=skeleton_mask)
                    fil.create_mask(verbose=False, use_existing_mask=True)
                    fil.medskel(verbose=False)
                    fil.analyze_skeletons(skel_thresh=50*u.pix)
                    
                    (accepted, statement) = MeasurerInstance.AssessFilament(fil, cap.get(1), dst, gradient / 255)
                    print(statement)
                    if not accepted:
                        continue
                    

                    ## idea for missing head length needed
                    # need to consider extent sitting outside of boundary

                
                # img = cv2.putText(dst, str(cap.get(1)), (100, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 255, 255), lineType=cv2.LINE_AA)
                # temp = cv2.resize(img, None, fy=0.7, fx=0.7)
                # cv2.imshow('binarized', temp)    
                
                images.append(opening)
                pos_frame = cap.get(1)
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_FRAME_COUNT, pos_frame-1)
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(1) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                print("\nFinal: " + str(MeasurerInstance.current_best["curvature"]) + "; " + str(MeasurerInstance.current_best["length"]) + "; " + str(MeasurerInstance.current_best["frame"]))
                break
            
        return images
        
    def AssessFilament(fil, frame, dst, gradient):
        filament = None
        try:
            if len(fil.lengths()) > 1:
                lengths = [q.value for q in fil.lengths()]
                index = lengths.index(max(fil.lengths()).value)
                filament = fil.filaments[index]
            else:
                filament = fil.filaments[0]
        except:
            return (False, "could not grab filament")
            
        long_path = fil.skeleton_longpath
        fil_length = filament.length(u.pix).value
        
        filament.rht_analysis()
        fil_curvature = filament.curvature.value
        fil_orientation = filament.orientation.value #* 180 / math.pi
        
        # run the intersection exercise for each end point
        longpath_pixel_coords = filament.longpath_pixel_coords
        end_pts = [(longpath_pixel_coords[0][0], longpath_pixel_coords[1][0]),
                   (longpath_pixel_coords[0][-1], longpath_pixel_coords[1][-1])]
        
        dimensions = np.shape(dst)
        
        # Convert orientation to slope
        slope = None
        if fil_orientation > 0:
            slope = math.pi / 2 - fil_orientation
        elif fil_orientation < 0:
            slope = - math.pi / 2 - fil_orientation
                
        for point in end_pts:
            # Create line mask and get intersecting points
            line_mask = np.zeros(dimensions)
            b = dimensions[0] - point[0] - slope * (point[1])
            
            for x in range(dimensions[1]):
                y = round(slope * x + b)
                if y < dimensions[0] - 1 and y >= 0:
                    line_mask[dimensions[0] - 1 - y, x] = 1
                    
            combined_array = line_mask + gradient
            pts_of_interest = np.where(combined_array > 1)
            
            # Get minimum distance from both ends to contour and add to the filament length
            minimum_distance = None
            min_point_set = None
            for i in range(len(pts_of_interest[0])):
                coord = np.array([pts_of_interest[0][i], pts_of_interest[1][i]])
                dist = np.linalg.norm(coord - np.array(point))
                # print("int: " + str(coord), "endpt: " + str(point), "dist: " + str(dist))
                if dist < 0: # endpoint on edge of a multi-pixel contour
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
                
            # Fill in the line mask array with extensions
            for x in range(point[1], min_point_set[1]):
                y = round(slope * x + b)
                if y < dimensions[0] - 1 and y >= 0:
                    long_path[dimensions[0] - 1 - y, x] = 1
                
        # endptImage = cv2.addWeighted(blendedImage * 255,0.5,gradient*255,0.5,0)
        # temp = cv2.resize(temp, None, fy=0.4, fx=0.4)
        # cv2.imshow('thing', temp) 
        # cv2.waitKey(0)
            
        print(slope, fil_curvature, "fil length: " + str(fil_length))
            
            
        # plt.imshow(filament.skeleton(pad_size=10,out_type='longpath'))
            
        if not any(MeasurerInstance.current_best):
            if fil_curvature < MeasurerInstance.max_curvature:
                MeasurerInstance.current_best = {"curvature": fil_curvature, "length": fil_length, "frame": frame, "image": None}
                image = MeasurerInstance.ShowImage(dst, (long_path * 255).astype('uint8'), resize=0.4, name="binarized") 
                return (True, "candidate accepted")
            else:
                return (False, "candidate too curved")

        elif fil_length <= MeasurerInstance.current_best["length"]:
            return (False, "filament too short")
        else:
            if fil_curvature < MeasurerInstance.max_curvature:
                print("candidate accepted")
                image = MeasurerInstance.ShowImage(dst, (long_path * 255).astype('uint8'), resize=0.4, name="binarized") 
                
                MeasurerInstance.current_best["curvature"] = fil_curvature
                MeasurerInstance.current_best["length"] = fil_length
                MeasurerInstance.current_best["image"] = image
                MeasurerInstance.current_best["frame"] = frame
                return (True, "candidate accepted")
            else:
                return (False, "candidate too curved")

    def ShowImage(image1, image2, resize=0.5, name='Image'):
        image = cv2.addWeighted(image1,0.5,image2,0.5,0)
        temp = cv2.resize(image, None, fy=resize, fx=resize)
        cv2.imshow(name, temp) 
        return image
    

measurer = MeasurerInstance()
        
        
        
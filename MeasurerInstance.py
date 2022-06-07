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
                
                if cap.get(1) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.4:
                    print("frame: " + str(cap.get(1)))
                    skeleton_mask = cv2.ximgproc.thinning(opening)
                    # https://stackoverflow.com/questions/53481596/python-image-finding-largest-branch-from-image-skeleton
                    fil = FilFinder2D(skeleton_mask, distance=1500*u.pix, mask=skeleton_mask)
                    fil.create_mask(verbose=False, use_existing_mask=True)
                    fil.medskel(verbose=False)
                    fil.analyze_skeletons(skel_thresh=50*u.pix)
                    
                    (accepted, statement) = MeasurerInstance.AssessFilament(fil, cap.get(1), dst)
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
        
    def AssessFilament(fil, frame, dst):
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

        # plt.imshow(filament.skeleton(pad_size=10,out_type='longpath'))
            
        # print("extents: " + str(fil.filament_extents))
        # print("endpts: " + str(filament.end_pts))
        # print("length: " + str(filament.length(u.pix)))
        # print("LPPC: " + str(filament.longpath_pixel_coords))

        if not any(MeasurerInstance.current_best):
            filament.rht_analysis()
            fil_curvature = filament.curvature.value
            print(filament.orientation.value * 180 / math.pi, fil_curvature, fil_length)
            
            if fil_curvature < MeasurerInstance.max_curvature:
                MeasurerInstance.current_best = {"curvature": fil_curvature, "length": fil_length, "frame": frame, "image": None}
            
                skeleton_Mat = (long_path * 255).astype('uint8')
                dst = cv2.addWeighted(dst,0.5,skeleton_Mat,0.5,0)
                temp = cv2.resize(dst, None, fy=0.7, fx=0.7)
                cv2.imshow('binarized', temp) 
                return (True, "candidate accepted")
            else:
                return (False, "candidate too curved")

        elif fil_length <= MeasurerInstance.current_best["length"]:
            return (False, "filament too short")
        else:
            filament.rht_analysis()
            fil_curvature = filament.curvature.value
            print(filament.orientation.value * 180 / math.pi, fil_curvature, fil_length)
            
            if fil_curvature < MeasurerInstance.max_curvature:
                print("candidate accepted")
                filament.plot_rht_distrib()
                skeleton_Mat = (long_path * 255).astype('uint8')
                dst = cv2.addWeighted(dst,0.5,skeleton_Mat,0.5,0)
                temp = cv2.resize(dst, None, fy=0.7, fx=0.7)
                cv2.imshow('binarized', temp)  
                
                MeasurerInstance.current_best["curvature"] = fil_curvature
                MeasurerInstance.current_best["length"] = fil_length
                MeasurerInstance.current_best["image"] = temp
                MeasurerInstance.current_best["frame"] = frame
                return (True, "candidate accepted")
            else:
                return (False, "candidate too curved")


measurer = MeasurerInstance()
        
        
        
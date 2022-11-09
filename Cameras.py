from tkinter import *

import pypylon
import pathlib
from pypylon import pylon
from pypylon import genicam
from pypylon import _genicam
from pypylon import _pylon

from datetime import datetime, time
from PIL import Image, ImageTk
import cv2
import time
import threading
import statistics

class Cameras():
    currentCam = None
    global current_frame
    global binarized_frame
    global raw_frame
    slope_list = []
    intercept_list = []
    
    def __init__(self, **kwargs):
        # converter for opencv bgr format
        Cameras.converter = pylon.ImageFormatConverter()
        Cameras.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        Cameras.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        # Status variables
        global current_frame
        global binarized_frame
        global raw_frame
        
        current_frame = None
        raw_frame = None
        binarized_frame = None
        
        Cameras.new_frame = False
        Cameras.framerate = 30
        Cameras.number_of_frames = 3
        Cameras.active_measurer = None
        Cameras.lock = threading.Lock()
        
        # Get the available cameras
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            Cameras.connected = False # will automatically shut down app in constructor
        else:
            ## Create and attach all Pylon Devices.
            try:
                Cameras.cameras = pylon.InstantCameraArray(len(devices))
                for i, cam in enumerate(Cameras.cameras):
                    cam.Attach(tlFactory.CreateDevice(devices[i]))
                    
                Cameras.currentCam = Cameras.cameras[0]
                Cameras.connected = True
                Cameras.StartGrabbing()
            except:
                Cameras.connected = False
        
    def ConnectMeasurer(measurer):
        Cameras.active_measurer = measurer
        print("connected measurer")
    
    def DisconnectMeasurer():
        Cameras.active_measurer = None
        print("disconnect")
        
    def StopGrabbing():
        try:
            if Cameras.currentCam.IsGrabbing():
                Cameras.currentCam.StopGrabbing()
                Cameras.currentCam.Close()
                
            return True
        except:
            return False
            
    def StartGrabbing():
        Cameras.currentCam.Open()
        Cameras.currentCam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        x = threading.Thread(target=Cameras.GrabLoop, daemon=True)
        x.start()
    
    def GrabLoop():
        while Cameras.currentCam.IsGrabbing():
            try:
                grabResult = Cameras.currentCam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            except genicam.GenericException as e:
                if ("Device has been removed from the PC" in str(e) or "No grab result data is referenced" in str(e)):
                    print(str(e))
                    state = Cameras.StopGrabbing()
                    Cameras.connected = False
                    
            # Image grabbed successfully?
            if grabResult is not None:
                if grabResult.GrabSucceeded():
                    image = Cameras.converter.Convert(grabResult)
                    img = image.GetArray()
                    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    grabResult.Release()
                    
                    Cameras.SetNewFrame(frame)
        
    def ChangeCamera(newCam):
        state = Cameras.StopGrabbing()
        if state:
            Cameras.currentCam = newCam
            try:
                Cameras.StartGrabbing()
            except Exception as e:
                if "Failed to open device" in str(e):
                    Cameras.connected = False
        else:
            print("ERROR")
            Cameras.connected = False # will automatically shut down app in constructor
    
    def TriggerAnalysis():
        Cameras.active_measurer.Analyze(Cameras.GetFixedNumFrames(Cameras.number_of_frames))
    
    def SetNewFrame(frame):
        global current_frame
        global binarized_frame
        global raw_frame
        
        Cameras.lock.acquire()
        raw_frame = frame
        if Cameras.active_measurer is not None and Cameras.active_measurer.background_is_trained:
            binarized_frame = Cameras.active_measurer.SubtractBackground(raw_frame)
            current_frame = binarized_frame if binarized_frame is not None else frame
        else:
            binarized_frame = None
            current_frame = frame
        
        Cameras.new_frame = True
        Cameras.lock.release()
            
    def GetFixedNumFrames(images_to_grab):
        global binarized_frame
        global raw_frame
        
        raw_list = []
        binarized_list = []
        while len(raw_list) < images_to_grab:
            Cameras.lock.acquire()
            
            if Cameras.new_frame:
                Cameras.new_frame = False
                raw_list.append(raw_frame)
                binarized_list.append(binarized_frame)
                
            Cameras.lock.release()
                
            print("{0} out of {1:.0f}".format(len(raw_list), images_to_grab))
            time.sleep(1/Cameras.framerate)
            
        return (raw_list, binarized_list)
    
    @staticmethod
    def ConvertPixelsToLength(pixels):
        return Cameras.GetSlope() * pixels + Cameras.GetIntercept()

    @staticmethod
    def ConvertLengthToPixels(length):
        return (length - Cameras.GetIntercept()) / Cameras.GetSlope()
    
    @staticmethod
    def GetSlope():
        if Cameras.slope_list:
            return statistics.mean(Cameras.slope_list)
        else:
            return None
    
    @staticmethod
    def GetIntercept():
        if Cameras.intercept_list:
            return statistics.mean(Cameras.intercept_list)
        else:
            return None
    
    def UpdateCamera(fishID=None, addText=None):
        if current_frame is not None:
            img = current_frame
            
            # ArUco markers
            # https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
            arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            arucoParams = cv2.aruco.DetectorParameters_create()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict,
                parameters=arucoParams)
            
            # verify *at least* one ArUco marker was detected
            if len(corners) > 0:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                
                distances = []
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned in
                    # top-left, top-right, bottom-right, and bottom-left order)
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                
                    # draw the bounding box of the ArUCo detection
                    cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)
                    
                    distance = ((topRight[0] - topLeft[0])**2 + (topRight[1] - topLeft[1])**2)**(0.5)
                    distances.append(distance)

                # Calculate calibration parameters
                if len(ids) > 1:
                    max_pixel_dist = max(distances)
                    min_pixel_dist = min(distances)
                    max_id = max(ids)
                    min_id = min(ids)
                    
                    temp_slope = (max_id-min_id) / (max_pixel_dist-min_pixel_dist) 
                    Cameras.slope_list.append(temp_slope)
                    
                    temp_intercept = max_id-temp_slope*max_pixel_dist
                    Cameras.intercept_list.append(temp_intercept)
            
            img = cv2.resize(img, None, fy=.39, fx=.39)
            
            img = cv2.putText(img, datetime.now().strftime("%d.%m.%Y %H:%M:%S"), (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)
            
            if fishID != None and fishID != '':
                img = cv2.putText(img, "Fish ID: " + fishID, (15, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)

            if addText != None and addText != '':
                text = addText
                y0, dy = 95, 25
                for i, line in enumerate(text.split('\n')):
                    y = y0 + i*dy
                    cv2.putText(img, line, (15, y), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)
            
            # Update the image to tk...
            img_update = ImageTk.PhotoImage(Image.fromarray(img))
            return img_update
        else:
            return None
        
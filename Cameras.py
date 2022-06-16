from tkinter import *
from pypylon import pylon
from datetime import datetime, time
from PIL import Image, ImageTk
import cv2
import time
import threading

class Cameras():
    currentCam = None
    global current_frame
    
    def __init__(self, **kwargs):
        # converter for opencv bgr format
        Cameras.converter = pylon.ImageFormatConverter()
        Cameras.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        Cameras.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        
        # Status variables
        global current_frame
        current_frame = None
        Cameras.new_frame = False
        Cameras.framerate = 30
        Cameras.active_measurer = None
        
        # Get the available cameras
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            Cameras.connected = False # will automatically shut down app in constructor
        else:
            ## Create and attach all Pylon Devices.
            Cameras.cameras = pylon.InstantCameraArray(len(devices))
            for i, cam in enumerate(Cameras.cameras):
                cam.Attach(tlFactory.CreateDevice(devices[i]))
                
            Cameras.currentCam = Cameras.cameras[0]
            Cameras.connected = True
            Cameras.StartGrabbing()
    
    def ApplySettings(measurer, **kwargs):
        Cameras.currentCam.ExposureTime.SetValue(float(kwargs["exposure"]))
        Cameras.currentCam.GainAuto.SetValue(kwargs["gain"])
        Cameras.currentCam.AcquisitionFrameRateEnable.SetValue(True)
        Cameras.currentCam.AcquisitionFrameRate.SetValue(float(kwargs["framerate"]))
        Cameras.framerate = float(kwargs["framerate"])
        
    def ConnectMeasurer(measurer):
        Cameras.active_measurer = measurer
    
    def DisconnectMeasurer():
        Cameras.active_measurer = None
        
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
            grabResult = Cameras.currentCam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                image = Cameras.converter.Convert(grabResult)
                img = image.GetArray()
                frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                
                Cameras.SetNewFrame(frame)
        
    def ChangeCamera(newCamera):
        state = Cameras.StopGrabbing()
        if state:
            Cameras.currentCam = newCamera
            Cameras.StartGrabbing()
        else:
            print("ERROR")
            Cameras.connected = False # will automatically shut down app in constructor
    
    def TriggerSkeletonize():
        global current_frame
        Cameras.active_measurer.SkeletonizeFrames(Cameras.GetFixedNumFrames(2))
    
    def SetNewFrame(frame):
        global current_frame
        if Cameras.active_measurer is not None and Cameras.active_measurer.background is not None:
            current_frame = Cameras.active_measurer.ProcessImage(frame)
        else:
            current_frame = frame
        
        Cameras.new_frame = True
            
    def GetFixedNumFrames(images_to_grab):
        image_list = []
        global current_frame
        
        while len(image_list) < images_to_grab:
            if Cameras.new_frame:
                Cameras.new_frame = False
                image_list.append(current_frame)
            time.sleep(1/Cameras.framerate)
        
        return image_list

    def UpdateCamera(fishID=None, addText=None):
        if current_frame is not None:
            img = cv2.resize(current_frame, None, fy=.39, fx=.39)
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
        
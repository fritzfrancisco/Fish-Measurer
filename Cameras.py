from tkinter import *
from pypylon import pylon
from datetime import datetime, time
from PIL import Image, ImageTk
import cv2
import sys

class Cameras():
    
    def __init__(self, **kwargs):
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
            self.ChangeCamera(Cameras.currentCam)
        
        # converter for opencv bgr format
        Cameras.converter = pylon.ImageFormatConverter()
        Cameras.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        Cameras.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    def ApplySettings(**kwargs):
        Cameras.currentCam.ExposureTime.SetValue(float(kwargs["exposure"]))
        Cameras.currentCam.GainAuto.SetValue(kwargs["gain"])
        Cameras.currentCam.AcquisitionFrameRateEnable.SetValue(True)
        Cameras.currentCam.AcquisitionFrameRate.SetValue(float(kwargs["framerate"]))
        Cameras.images_to_grab = kwargs["framerate"] * kwargs["duration"]
    
    def ChangeCamera(self, newCamera):
        if Cameras.currentCam.IsGrabbing():
            Cameras.currentCam.StopGrabbing()
            Cameras.currentCam.Close()
                
        Cameras.currentCam = newCamera
        Cameras.currentCam.Open()
        Cameras.currentCam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def UpdateCamera(fishID=None, addText=None):
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = Cameras.currentCam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            image = Cameras.converter.Convert(grabResult)
            img = image.GetArray()
            img = cv2.resize(img, None, fy=.39, fx=.39)
            
            # Apply text to image
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
            frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_update = ImageTk.PhotoImage(Image.fromarray(frame))
            
            k = cv2.waitKey(1)
            if k == 27:
                Cameras.currentCam.StopGrabbing()
                Cameras.currentCam.Close()
                cv2.destroyAllWindows()
            
            return img_update
        else:
            return None
        
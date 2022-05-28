import tkinter as tk
from datetime import datetime, time
from tkinter import *
from tkinter import filedialog
import os
from turtle import bgcolor, width
from typing import Dict
from pypylon import pylon
import json
import cv2

# Get the available cameras
tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RuntimeException("No camera present.")

## Create and attach all Pylon Devices.
cameras = pylon.InstantCameraArray(len(devices))
for i, cam in enumerate(cameras):
    cam.Attach(tlFactory.CreateDevice(devices[i]))
    
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
currentCam = cameras[0]
currentCam.Open()
currentCam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
cv2.namedWindow(str('Basler Capture ' + str(currentCam.GetDeviceInfo().GetSerialNumber())), cv2.WINDOW_NORMAL)

while currentCam.IsGrabbing():
    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
    grabResult = currentCam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    # Image grabbed successfully?
    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        
        img = cv2.putText(img, datetime.now().strftime("%Y%m%d %H:%M:%S"), (
            15, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        
        cv2.imshow(str('Basler Capture ' +
                       str(currentCam.GetDeviceInfo().GetSerialNumber())), img)
        k = cv2.waitKey(1)
        if k == 27:
            currentCam.StopGrabbing()
            currentCam.Close()
            cv2.destroyAllWindows()
            break
        
    else:
        print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
    
    

currentCam.Close()
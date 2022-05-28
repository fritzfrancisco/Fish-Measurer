import tkinter as tk
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import cv2
from datetime import datetime, time
from turtle import bgcolor, width
from typing import Dict
from pypylon import pylon

# Get the available cameras
tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()
if len(devices) == 0:
    raise pylon.RuntimeException("No camera present.")

## Create and attach all Pylon Devices.
cameras = pylon.InstantCameraArray(len(devices))
for i, camz in enumerate(cameras):
    camz.Attach(tlFactory.CreateDevice(devices[i]))
    
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
global currentCam
currentCam = cameras[0]
currentCam.Open()
currentCam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

## TKITNER
ikkuna=tk.Tk()
ikkuna.title("Example about handy CV2 and tk combination...")

global frame

paneeli_image=tk.Label(ikkuna) #,image=img)
paneeli_image.grid(row=0,column=0,columnspan=3,pady=1,padx=10, sticky='nsew')

message="You can see some \nclassification results \nhere after you add some intelligent  \nadditional code to your combined and handy \n tk & CV2 solution!"
paneeli_text=tk.Label(ikkuna,text=message)
paneeli_text.grid(row=1,column=1,pady=1,padx=10)

def otakuva():
    global frame
    global currentCam
    
    while currentCam.IsGrabbing():
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = currentCam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            image = converter.Convert(grabResult)
            img = image.GetArray()
            
            # ikkuna.update()
            # print(paneeli_image.winfo_width())
            # print(paneeli_image.winfo_height())
            img = cv2.resize(img, None, fy=.3, fx=.3)
            
            img = cv2.putText(img, datetime.now().strftime("%Y%m%d %H:%M:%S"), (
                15, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            
            #Update the image to tk...
            frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_update = ImageTk.PhotoImage(Image.fromarray(frame))
            paneeli_image.configure(image=img_update)
            paneeli_image.image=img_update
            paneeli_image.update()
            
            k = cv2.waitKey(1)
            if k == 27:
                currentCam.StopGrabbing()
                currentCam.Close()
                cv2.destroyAllWindows()
                break
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
        
def lopeta():
    global currentCam
    currentCam.release()
    cv2.destroyAllWindows()
    print("Stopped!")

painike_korkeus=10
painike_1=tk.Button(ikkuna,text="Start",command=otakuva,height=5,width=20)
painike_1.grid(row=1,column=0,pady=10,padx=10)
painike_1.config(height=1*painike_korkeus,width=20)

painike_korkeus=10
painike_1=tk.Button(ikkuna,text="Stop",command=lopeta,height=5,width=20)
painike_1.grid(row=1,column=2,pady=10,padx=10)
painike_1.config(height=1*painike_korkeus,width=20)

ikkuna.mainloop()
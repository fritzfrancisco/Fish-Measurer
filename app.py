import tkinter as tk
from tkinter import *
from tkinter import filedialog
import os
from turtle import bgcolor, width
from typing import Dict
from pypylon import pylon
from datetime import datetime, time
import json
from PIL import Image, ImageTk
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

global currentCam
currentCam = cameras[0]
currentCam.Open()
currentCam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    
def UpdateCamera():
    global currentCam
    global cameras
    global frame
    
    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
    grabResult = currentCam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    # Image grabbed successfully?
    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        
        img = cv2.resize(img, None, fy=.39, fx=.39)
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
    else:
        print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
    
    paneeli_image.after(15, UpdateCamera)
    

# Create the app
rootWindow = tk.Tk()
rootWindow.geometry("1422x800")
rootWindow.title('Fish Measurer')
rootWindow.columnconfigure(0)
rootWindow.columnconfigure(1, weight=1)

settingsFrame = tk.Frame(relief='flat')
settingsFrame.configure(background="grey80")
settingsFrame.grid(row = 0, column = 0, sticky='ns')

## -- VIDEO -------------------------------------------------------------------------------------------
videoFrame = tk.Frame(master=rootWindow, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey5")
videoFrame.grid(row=0, column=1, sticky="nsew")

paneeli_image=tk.Label(videoFrame) #,image=img)
paneeli_image.pack(fill=tk.BOTH, expand=True)

# converter for opencv bgr format
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            
## -- CAMERA SETTINGS --------------------------------------------------------------------------------
cameraFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=2, pady=10, bg="grey80")
cameraFrame.pack(fill=tk.X)

# Camera settings title
cameraHeaderFrame = tk.Frame(relief='groove', borderwidth=2, padx=5, pady=5, bg="grey25", master=cameraFrame)
cameraHeaderFrame.pack(fill=tk.X)

cameraHeaderText = tk.Label(text="CAMERA SETTINGS", master=cameraHeaderFrame, background="grey25", fg="white")
cameraHeaderText.config(font=("Courier", 24))
cameraHeaderText.pack(fill=tk.X)

# Camera-select dropdown
dropdownFrame = tk.Frame(master=cameraFrame, relief='flat', borderwidth=2, pady=5, bg="grey80")
dropdownFrame.pack(fill=tk.X)

manualFillPrompt = tk.Label(text="Choose camera:", master=dropdownFrame, bg="grey80", anchor="w")
manualFillPrompt.config(font=("Courier", 10))
manualFillPrompt.pack(fill=tk.X)

def ChangeCamera(newCam):
    global currentCam
    currentCam.StopGrabbing()
    currentCam.Close()
    print("from: " + currentCam + "; to: " + newCam)
    
    currentCam = newCam
    currentCam.Open()
    currentCam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

camsDict = {}
CAMOPTIONS = []
for i, cam in enumerate(cameras):
    CAMOPTIONS.append(str(cam.GetDeviceInfo().GetModelName()))
    camsDict[i] = cam

camSelectVariable = StringVar()
camSelectVariable.set(CAMOPTIONS[0]) # default value
camSelectVariable.trace_add('write', lambda *args, newCam = camsDict[CAMOPTIONS.index(camSelectVariable.get())]: ChangeCamera)
w = OptionMenu(dropdownFrame, camSelectVariable, *CAMOPTIONS)
w["highlightthickness"] = 0
w.pack(fill=tk.X)

# Camera settings
cameraSettingsFrame = tk.Frame(master=cameraFrame, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey60")
cameraSettingsFrame.pack(fill=tk.X)

# dictionary = {
#     "exposure" : 2000,
#     "gain" : "2",
#     "duration" : 8.6,
#     "framerate" : 44,
#     "threshold" : 150
# }

# with open('result.json', 'w') as fp:
#     json.dump(dictionary, fp)

def UploadAction():
    filename = filedialog.askopenfilename()
    
    if filename:
        # Opening JSON file
        with open(filename) as json_file:
            data = json.load(json_file)
            
            uploadFailed = []
            if "exposure" in data:
                exposureEntry.delete(0, END)
                exposureEntry.insert(0, data["exposure"]) 
            else:
                uploadFailed.append("exposure")
                
            if "gain" in data:
                if data["gain"] in GAINOPTIONS:
                    gainVariable.set(GAINOPTIONS[GAINOPTIONS.index(data["gain"])])
                else:
                    uploadFailed.append("gain")
            else:
                uploadFailed.append("gain")
                
            if "duration" in data:
                durationEntry.delete(0, END)
                durationEntry.insert(0, data["duration"]) 
            else:
                uploadFailed.append("duration")
                
            if "framerate" in data:
                framerateEntry.delete(0, END)
                framerateEntry.insert(0, data["framerate"]) 
            else:
                uploadFailed.append("framerate")
                
            if "threshold" in data:
                thresholdValueEntry.delete(0, END)
                thresholdValueEntry.insert(0, data["threshold"]) 
            else:
                uploadFailed.append("threshold")
            
            if uploadFailed:
                tk.messagebox.showerror("Config File Errors", "The following parameters were not updated due to parsing errors:\n\n" + str(uploadFailed))

button = tk.Button(cameraSettingsFrame, text='Upload .config file', command=UploadAction)
button.pack(fill=tk.X)

manualFillPrompt = tk.Label(text="Or enter settings manually:", master=cameraSettingsFrame, bg="grey60", pady=10)
manualFillPrompt.config(font=("Courier", 12))
manualFillPrompt.pack(fill=tk.X)

inputsFrame = tk.Frame(master=cameraSettingsFrame, relief='flat', borderwidth=2)
inputsFrame.pack(fill=tk.X)
inputsFrame.columnconfigure(0)
inputsFrame.columnconfigure(1, weight=1)

# Settings inputs
exposureText = tk.Label(text="Exposure (ms): ", master=inputsFrame, pady=3)
exposureEntry = tk.Entry(inputsFrame, justify='center')
exposureEntry.insert(0, "50000") 
exposureText.grid(row = 0, column = 0, sticky='w', padx=5)
exposureEntry.grid(row = 0, column = 1, sticky='ew', padx=5)

gainSetting = tk.Label(text="Gain Setting: ", master=inputsFrame)
GAINOPTIONS = [
"Once",
"Continuous",
"Off"
]
gainVariable = StringVar()
gainVariable.set(GAINOPTIONS[0]) # default value
w = OptionMenu(inputsFrame, gainVariable, *GAINOPTIONS)
w["highlightthickness"] = 0
w.grid(row = 1, column = 1, sticky='ew', padx=5)
gainSetting.grid(row = 1, column = 0, sticky='w', padx=5)

duration = tk.Label(text="Duration (s): ", master=inputsFrame, pady=3)
durationEntry = tk.Entry(inputsFrame, justify='center') 
durationEntry.insert(0, "5") 
duration.grid(row = 2, column = 0, sticky='w', padx=5)
durationEntry.grid(row = 2, column = 1, sticky='ew', padx=5)

framerate = tk.Label(text="Framerate (fps): ", master=inputsFrame, pady=3)
framerateEntry = tk.Entry(inputsFrame, justify='center') 
framerateEntry.insert(0, "30") 
framerate.grid(row = 3, column = 0, sticky='w', padx=5)
framerateEntry.grid(row = 3, column = 1, sticky='ew', padx=5)

thresholdValue = tk.Label(text="Threshold [0, 255]: ", master=inputsFrame, pady=3)
thresholdValueEntry = tk.Entry(inputsFrame, justify='center') 
thresholdValueEntry.insert(0, "100") 
thresholdValue.grid(row = 4, column = 0, sticky='w', padx=5)
thresholdValueEntry.grid(row = 4, column = 1, sticky='ew', padx=5)

## -- OUTPUT SETTINGS -----------------------------------------------------------------------------------------------
outputFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=2, pady=10, bg="grey80")
outputFrame.pack(fill=tk.BOTH)

# Camera settings title
outputHeaderFrame = tk.Frame(relief='groove', borderwidth=2, padx=5, pady=5, bg="grey25", master=outputFrame)
outputHeaderFrame.pack(fill=tk.X)

outputHeaderText = tk.Label(text="OUTPUT SETTINGS", master=outputHeaderFrame, background="grey25", fg="white")
outputHeaderText.config(font=("Courier", 24))
outputHeaderText.pack(fill=tk.X)

# Output settings
outputSettingsFrame = tk.Frame(master=outputFrame, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey80")
outputSettingsFrame.pack(fill=tk.X)

# Output-destination-select dropdown
destinationPrompt = tk.Label(text="Choose output folder:", master=outputSettingsFrame, bg="grey80", anchor="w")
destinationPrompt.config(font=("Courier", 10))
destinationPrompt.pack(fill=tk.X)

browseFrame = tk.Frame(master=outputSettingsFrame, relief='flat', borderwidth=2, bg="grey80")
browseFrame.columnconfigure(0)
browseFrame.columnconfigure(1)
browseFrame.pack(fill=tk.X)

def BrowseButton():
    global folder_path
    folderName = filedialog.askdirectory()
    folder_path.set(folderName)
    print(folderName)
    
button = tk.Button(browseFrame, text='Browse...', command=BrowseButton)
button.grid(row=0, column=0, sticky="nsew")

folder_path = StringVar()
folder_path.set(os.getcwd())
selectedFolder = Label(master=browseFrame, textvariable=folder_path, width=31, anchor='e')
selectedFolder.grid(row=0, column=1, sticky="nsew")

# File format selector
formatFrame = tk.Frame(master=outputSettingsFrame, relief='flat', borderwidth=2, bg="grey80")
formatFrame.columnconfigure(0)
formatFrame.columnconfigure(1, weight=1)
formatFrame.pack(fill=tk.X)

saveFormatPrompt = tk.Label(text="Select output format:", master=formatFrame, bg="grey80", anchor="w")
saveFormatPrompt.config(font=("Courier", 10))
saveFormatPrompt.grid(row = 0, column = 0, sticky='w')

OPTIONS = [
".jpeg",
".png",
".tiff"
]
outputFormatVariable = StringVar()
outputFormatVariable.set(OPTIONS[0]) # default value
w = OptionMenu(formatFrame, outputFormatVariable, *OPTIONS)
w["highlightthickness"] = 0
w.grid(row = 0, column = 1, sticky='ew', padx=5, pady=5)

# Fish ID and freetext
watermarkFrame = tk.Frame(master=outputSettingsFrame, relief='flat', borderwidth=2, bg="grey60", padx=5, pady=2)
watermarkFrame.pack(fill=tk.X)

## Fish ID
fishIDFrame = tk.Frame(master=watermarkFrame, relief='flat', borderwidth=2)
fishIDFrame.columnconfigure(0)
fishIDFrame.columnconfigure(1, weight=1)
fishIDFrame.pack(fill=tk.X, pady=3)

fishID = tk.Label(text="Fish ID: ", master=fishIDFrame, pady=3)
fishIDEntry = tk.Entry(fishIDFrame, justify='left') 
fishID.grid(row = 0, column = 0, sticky='w', padx=5)
fishIDEntry.grid(row = 0, column = 1, sticky='ew', padx=5)

## Freetext
freetextFrame = tk.Frame(master=watermarkFrame, relief='flat', borderwidth=2, padx=5)
freetextFrame.pack(fill=tk.BOTH, pady=4)

freetextPrompt = tk.Label(text="Additional text for watermark:", master=freetextFrame, anchor="w")
freetextPrompt.config(font=("Courier", 10))
freetextPrompt.pack(fill=tk.X)

additionalText = tk.Text(freetextFrame, height=5, width=5)
additionalText.pack(fill=tk.X, pady=5)

## -- START BUTTON -----------------------------------------------------------------------------------
startFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=10, pady=10, bg="grey80")
startFrame.pack(fill=tk.BOTH)

def StartButton(event=None):
    print("hi")
    
startButton = tk.Button(startFrame, text='START', command=StartButton, bg="red", font=("Courier", 24), fg="white")
startButton.pack(fill=tk.BOTH)


paneeli_image.after(15, UpdateCamera)
rootWindow.mainloop()
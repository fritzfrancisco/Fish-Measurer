import tkinter as tk
from tkinter import *
from tkinter import filedialog
import os
from tkinter.font import BOLD
from turtle import bgcolor, width

from cv2 import threshold

rootWindow = tk.Tk()
rootWindow.geometry("1422x800")
rootWindow.title('Fish Measurer')
rootWindow.columnconfigure(0)
rootWindow.columnconfigure(1, weight=1)

settingsFrame = tk.Frame(relief='flat')
settingsFrame.configure(background="grey80")

# canvas = tk.Canvas(settingsFrame)

# scrollbar  = Scrollbar(settingsFrame, command=canvas.yview)
# scrollbar.pack(side = RIGHT, fill = Y)
# scrollable_frame = tk.Frame(canvas)


settingsFrame.grid(row = 0, column = 0, sticky='ns')

## -- CAMERA SETTINGS --------------------------------------------------------------------------------
cameraFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=5, pady=10, bg="grey80")
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

OPTIONS = [
"Camera-1",
"Camera-2",
"Camera-3"
]
variable = StringVar(rootWindow)
variable.set(OPTIONS[0]) # default value
w = OptionMenu(dropdownFrame, variable, *OPTIONS)
w["highlightthickness"] = 0
w.pack(fill=tk.X)

# Camera settings
cameraSettingsFrame = tk.Frame(master=cameraFrame, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey60")
cameraSettingsFrame.pack(fill=tk.X)

def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)

button = tk.Button(cameraSettingsFrame, text='Upload .config file', command=UploadAction)
button.pack(fill=tk.X)

manualFillPrompt = tk.Label(text="Or enter settings manually:", master=cameraSettingsFrame, bg="grey60", pady=10)
manualFillPrompt.config(font=("Courier", 12))
manualFillPrompt.pack(fill=tk.X)

inputsFrame = tk.Frame(master=cameraSettingsFrame, relief='flat', borderwidth=2)
inputsFrame.pack(fill=tk.X)
inputsFrame.columnconfigure(0)
inputsFrame.columnconfigure(1, weight=1)

exposureText = tk.Label(text="Exposure (ms): ", master=inputsFrame, pady=3)
exposureEntry = tk.Entry(inputsFrame, justify='center')
exposureEntry.insert(0, "50000") 
exposureText.grid(row = 0, column = 0, sticky='w', padx=5)
exposureEntry.grid(row = 0, column = 1, sticky='ew', padx=5)

gainSetting = tk.Label(text="Gain Setting: ", master=inputsFrame)
OPTIONS = [
"Once",
"Continuous",
"Off"
]
variable = StringVar(rootWindow)
variable.set(OPTIONS[0]) # default value
w = OptionMenu(inputsFrame, variable, *OPTIONS)
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
outputFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=5, pady=10, bg="grey80")
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
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    
button = tk.Button(browseFrame, text='Browse...', command=BrowseButton)
button.grid(row=0, column=0, sticky="nsew")

folder_path = StringVar()
selectedFolder = Label(master=browseFrame, textvariable=folder_path, width=31)
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
variable = StringVar(rootWindow)
variable.set(OPTIONS[0]) # default value
w = OptionMenu(formatFrame, variable, *OPTIONS)
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
    filename = filedialog.askopenfilename()
    print('Selected:', filename)
startButton = tk.Button(startFrame, text='START', command=StartButton, bg="red", font=("Courier", 24), fg="white")
startButton.pack(fill=tk.BOTH)

## -- VIDEO -------------------------------------------------------------------------------------------
videoFrame = tk.Frame(master=rootWindow, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey5")
videoFrame.grid(row=0, column=1, sticky="nsew")

additionalText = tk.Text(videoFrame, height=120, width=120)
additionalText.pack(fill=tk.BOTH)

rootWindow.mainloop()
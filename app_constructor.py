import tkinter as tk
from tkinter import *
from tkinter import filedialog
import os
from datetime import datetime, time
from DigitEntry import DigitEntry
from MeasurerInstance import MeasurerInstance
from Cameras import Cameras
import sys
import threading


class ConstructApp():
    def __init__(self, **kwargs):
             
        # Create the app
        rootWindow = tk.Tk()
        rootWindow.geometry("1422x800")
        rootWindow.title('Fish Measurer')
        rootWindow.columnconfigure(0)
        rootWindow.columnconfigure(1, weight=1)
        
        if not Cameras.connected:
            tk.messagebox.showerror("Camera Connection Error", "The app cannot find a camera connected to this device or \
                could not connect to the selected camera. Please verify the device connection and try again.")
            sys.exit()
            
        settingsFrame = tk.Frame(relief='flat')
        settingsFrame.configure(background="grey80")
        settingsFrame.grid(row = 0, column = 0, sticky='ns')

        ## -- VIDEO -------------------------------------------------------------------------------------------
        videoFrame = tk.Frame(master=rootWindow, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey5")
        videoFrame.grid(row=0, column=1, sticky="nsew")

        paneeli_image=tk.Label(videoFrame) #,image=img)
        paneeli_image.pack(fill=tk.BOTH, expand=True)
        
        def UpdateFeed():
            fishID = ConstructApp.fishIDEntry.get()
            addText = ConstructApp.additionalText.get("1.0",'end-1c')
            
            # Update the measurer
            MeasurerInstance.fishID = fishID
            MeasurerInstance.addText = addText
            
            # Fish for the new image
            newImage = Cameras.UpdateCamera(fishID=fishID, addText=addText)
            if newImage != None:
                paneeli_image.configure(image=newImage)
                paneeli_image.image=newImage
                paneeli_image.update()
                paneeli_image.after(1/Cameras.framerate*1000, UpdateFeed)
            else:
                if not Cameras.connected:
                    tk.messagebox.showerror("Camera Connection Error", "The app cannot find a camera connected to this device. Please verify the connection and try again.")
                    sys.exit()
                else:
                    # print("image None")
                    paneeli_image.after(1/Cameras.framerate*1000, UpdateFeed)
                
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

        camsDict = {}
        CAMOPTIONS = []
        for i, cam in enumerate(Cameras.cameras):
            CAMOPTIONS.append(str(cam.GetDeviceInfo().GetModelName()))
            camsDict[i] = cam

        camSelectVariable = StringVar()
        camSelectVariable.set(CAMOPTIONS[0]) # default value
        camSelectVariable.trace_add('write', lambda *args, newCam = camsDict[CAMOPTIONS.index(camSelectVariable.get())]: Cameras.ChangeCamera)
        w = OptionMenu(dropdownFrame, camSelectVariable, *CAMOPTIONS)
        w["highlightthickness"] = 0
        w.pack(fill=tk.X)

        # Camera settings
        cameraSettingsFrame = tk.Frame(master=cameraFrame, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey60")
        cameraSettingsFrame.pack(fill=tk.X)

        manualFillPrompt = tk.Label(text="Or enter settings manually:", master=cameraSettingsFrame, bg="grey60", pady=10)
        manualFillPrompt.config(font=("Courier", 12))
        manualFillPrompt.pack(fill=tk.X)

        self.inputsFrame = tk.Frame(master=cameraSettingsFrame, relief='flat', borderwidth=2)
        self.inputsFrame.pack(fill=tk.X)
        self.inputsFrame.columnconfigure(0)
        self.inputsFrame.columnconfigure(1, weight=1)

        # Settings inputs
        ConstructApp.exposureSetting = DigitEntry("Exposure (ms): ", 10000, 0, self.inputsFrame, trace="exposure")

        gainSetting = tk.Label(text="Gain Setting: ", master=self.inputsFrame)
        GAINOPTIONS = [
        "Once",
        "Continuous",
        "Off"
        ]
        ConstructApp.gainVariable = StringVar()
        ConstructApp.gainVariable.set(GAINOPTIONS[0]) # default value
        ConstructApp.gainVariable.trace('w', SendGain)
        w = OptionMenu(self.inputsFrame, ConstructApp.gainVariable, *GAINOPTIONS)
        w["highlightthickness"] = 0
        w.grid(row = 1, column = 1, sticky='ew', padx=5)
        gainSetting.grid(row = 1, column = 0, sticky='w', padx=5)
        
        def SendGain():
            Cameras.currentCam.GainAuto.SetValue(float(ConstructApp.gainVariable.get()))

        ConstructApp.durationSetting = DigitEntry("Duration (s): ", 5, 2, self.inputsFrame) # don't need
        ConstructApp.frameRateSetting = DigitEntry("Framerate (fps): ", 30, 3, self.inputsFrame, trace="framerate")

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

        ConstructApp.folder_path = StringVar()
        ConstructApp.folder_path.set(os.getcwd())
        selectedFolder = Label(master=browseFrame, textvariable=ConstructApp.folder_path, width=31, anchor='e')
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
        ConstructApp.outputFormatVariable = StringVar()
        ConstructApp.outputFormatVariable.set(OPTIONS[0]) # default value
        w = OptionMenu(formatFrame, ConstructApp.outputFormatVariable, *OPTIONS)
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
        ConstructApp.fishIDEntry = tk.Entry(fishIDFrame, justify='left') 
        fishID.grid(row = 0, column = 0, sticky='w', padx=5)
        ConstructApp.fishIDEntry.grid(row = 0, column = 1, sticky='ew', padx=5)

        ## Freetext
        freetextFrame = tk.Frame(master=watermarkFrame, relief='flat', borderwidth=2, padx=5)
        freetextFrame.pack(fill=tk.BOTH, pady=4)

        freetextPrompt = tk.Label(text="Additional text for watermark:", master=freetextFrame, anchor="w")
        freetextPrompt.config(font=("Courier", 10))
        freetextPrompt.pack(fill=tk.X)

        ConstructApp.additionalText = tk.Text(freetextFrame, height=5, width=5)
        ConstructApp.additionalText.pack(fill=tk.X, pady=5)
        
        ## -- BG BUTTON -----------------------------------------------------------------------------------
        backgroundFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=10, pady=10, bg="grey80")
        backgroundFrame.pack(fill=tk.BOTH)
        
        self.backgroundButton = tk.Button(backgroundFrame, text='TRAIN', command=self.BackgroundButton, bg="DarkSeaGreen", font=("Courier", 24), fg="white")
        self.backgroundButton.pack(fill=tk.BOTH)

        ## -- START BUTTON -----------------------------------------------------------------------------------
        ConstructApp.thresholdSetting = DigitEntry("Threshold [0, 255]: ", 100, 4, settingsFrame, lower=0, upper=255, trace="threshold")
        
        startFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=10, pady=10, bg="grey80")
        startFrame.pack(fill=tk.BOTH)

        self.startButton = tk.Button(startFrame, text='START', command=self.StartButton, bg="red", font=("Courier", 24), fg="white", state="disabled")
        self.startButton.pack(fill=tk.BOTH)

        paneeli_image.after(15, UpdateFeed)
        rootWindow.mainloop()
        
    def ButtonTextCountDown(button, time):
        for i in range(time, 0, -1):
            button["text"] = str(i)
            time.sleep(1)
            
    def BackgroundButton(self):
        if self.backgroundButton["text"] == "TRAIN":
            # Locking
            self.backgroundButton["state"] = "disabled"
            for child in self.inputsFrame.winfo_children():
                child.configure(state='disable')
            
            measurer = MeasurerInstance(ConstructApp.folder_path.get(), ConstructApp.outputFormatVariable.get())
            
            # Thread the Tkinter button countdown
            x = threading.Thread(target=ConstructApp.ButtonTextCountDown, args=(self.backgroundButton, 10,), daemon=True)
            x.start()
            # if doesn't work, can place a timer in the TrainBackground method and put the for loop here instead of in the measurer
            
            # Train our background for later subtraction
            measurer.TrainBackground()
            
            # Reconfigure the buttons
            self.backgroundButton["text"] = "RESTART"
            self.backgroundButton.configure(bg = "red")
            self.backgroundButton["state"] == "normal"
            self.startButton.configure(bg = "DarkSeaGreen")
            self.startButton["state"] == "normal"
            
            ## trigger threshold camera view
            
        elif self.backgroundButton["text"] == "RESTART":
            # Reconfigure buttons and delete the previous instance
            self.backgroundButton["text"] = "TRAIN"
            self.backgroundButton.configure(bg = "DarkSeaGreen")
            self.startButton.configure(bg = "red")
            self.startButton["state"] == "disabled"
            
            # Unlock settings
            for child in self.inputsFrame.winfo_children():
                child.configure(state='normal')
            
            Cameras.DisconnectMeasurer()
            measurer = None
            
    def StartButton(self):
        self.backgroundButton["state"] == "disabled"
        self.startButton["state"] == "disabled"
        ConstructApp.thresholdSetting.Activate(False)
        
        Cameras.active_measurer.Skeletonize()
        
        self.startButton["state"] == "normal"
        self.backgroundButton["state"] == "normal"
        ConstructApp.thresholdSetting.Activate(True)
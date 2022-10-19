import tkinter as tk
from tkinter import *
from tkinter import filedialog
import os
from DigitEntry import DigitEntry
from MeasurerInstance import MeasurerInstance
from Cameras import Cameras
import math
import sys
import threading

class ConstructApp():
    def __init__(self, **kwargs):
        # Create the app
        rootWindow = tk.Tk()
        rootWindow.geometry("1133x850")
        rootWindow.title('Fish Measurer')
        rootWindow.columnconfigure(0)
        rootWindow.columnconfigure(1, weight=1)
        
        if not Cameras.connected:
            tk.messagebox.showerror("Camera Connection Error", "The app cannot find a camera connected to this device or could not connect to the selected camera (check that the USB cable/port is 3.0 compatible). Please verify the device connection and try again.")
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
            
            if not Cameras.connected:
                tk.messagebox.showerror("Camera Connection Error", "The app cannot find a camera connected to this device. Please verify the connection and try again.")
                sys.exit()
            
            # Fish for the new image
            newImage = Cameras.UpdateCamera(fishID=fishID, addText=addText)
            if newImage != None:
                paneeli_image.configure(image=newImage)
                paneeli_image.image=newImage
                paneeli_image.update()
                paneeli_image.after(math.ceil(1/Cameras.framerate*1000), UpdateFeed)
            else:
                if not Cameras.connected:
                    tk.messagebox.showerror("Camera Connection Error", "The app cannot find a camera connected to this device. Please verify the connection and try again.")
                    sys.exit()
                else:
                    # print("image None")
                    paneeli_image.after(math.ceil(1/Cameras.framerate*1000), UpdateFeed)
                
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
            CAMOPTIONS.append(str(i+1) + ": " + str(cam.GetDeviceInfo().GetModelName()) + "; S/N: " + str(cam.GetDeviceInfo().GetSerialNumber()))
            camsDict[i] = cam
        
        def SendCamChange(selection):
            Cameras.ChangeCamera(camsDict[int(selection[0])-1])

        camSelectVariable = StringVar()
        camSelectVariable.set(CAMOPTIONS[0]) # default value
        self.cam_choice_dropdown = OptionMenu(dropdownFrame, camSelectVariable, *CAMOPTIONS, command=SendCamChange)
        self.cam_choice_dropdown["highlightthickness"] = 0
        self.cam_choice_dropdown.pack(fill=tk.X)
        
        # Camera settings
        cameraSettingsFrame = tk.Frame(master=cameraFrame, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey60")
        cameraSettingsFrame.pack(fill=tk.X)

        self.inputsFrame = tk.Frame(master=cameraSettingsFrame, relief='flat', borderwidth=2)
        self.inputsFrame.pack(fill=tk.X)
        self.inputsFrame.columnconfigure(0)
        self.inputsFrame.columnconfigure(1, weight=1)

        # Settings inputs
        ConstructApp.exposureSetting = DigitEntry("Exposure (ms): ", 100, 0, self.inputsFrame, trace="exposure")
        
        def SendGain(selection):
            ## error handling for failure? Tkinter pop-up
            Cameras.currentCam.GainAuto.SetValue(selection)

        ## Gain parameter
        gainSetting = tk.Label(text="Gain Setting: ", master=self.inputsFrame)
        GAINOPTIONS = [
        "Once",
        "Continuous",
        "Off"
        ]
        ConstructApp.gainVariable = StringVar()
        ConstructApp.gainVariable.set(GAINOPTIONS[0]) # default value
        SendGain(GAINOPTIONS[0])
        w = OptionMenu(self.inputsFrame, ConstructApp.gainVariable, *GAINOPTIONS, command=SendGain)
        w["highlightthickness"] = 0
        w.grid(row = 1, column = 1, sticky='ew', padx=5)
        gainSetting.grid(row = 1, column = 0, sticky='w', padx=5)
        
        ## Gain parameter
        def SendWB(selection):
            ## error handling for failure? Tkinter pop-up
            Cameras.currentCam.BalanceWhiteAuto.SetValue(selection)

        white_balance_setting = tk.Label(text="White Balance: ", master=self.inputsFrame)
        WBOPTIONS = [
        "Once",
        "Continuous",
        "Off"
        ]
        ConstructApp.wb_variable = StringVar()
        ConstructApp.wb_variable.set(WBOPTIONS[2]) # default value
        SendWB(WBOPTIONS[2])
        w = OptionMenu(self.inputsFrame, ConstructApp.wb_variable, *WBOPTIONS, command=SendWB)
        w["highlightthickness"] = 0
        w.grid(row = 2, column = 1, sticky='ew', padx=5)
        white_balance_setting.grid(row = 2, column = 0, sticky='w', padx=5)
        
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
            folderName = filedialog.askdirectory()
            if folderName:
                ConstructApp.folder_path.set(folderName)
                print(folderName)
            
        ConstructApp.button = tk.Button(browseFrame, text='Browse...', command=BrowseButton)
        ConstructApp.button.grid(row=0, column=0, sticky="nsew")

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
        ConstructApp.w = OptionMenu(formatFrame, ConstructApp.outputFormatVariable, *OPTIONS)
        ConstructApp.w["highlightthickness"] = 0
        ConstructApp.w.grid(row = 0, column = 1, sticky='ew', padx=5, pady=5)

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
        
        self.backgroundButton = tk.Button(backgroundFrame, text='TRAIN', command=self.BackgroundButton, bg="#74B224", font=("Courier", 24), fg="white", disabledforeground="white")
        self.backgroundButton.pack(fill=tk.BOTH)

        ## -- START BUTTON -----------------------------------------------------------------------------------
        startFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=5, pady=5, bg="grey60")
        startFrame.pack(fill=tk.BOTH, padx=5)
        
        innerStartFrame = tk.Frame(master=startFrame, relief='flat', borderwidth=2, padx=5)
        innerStartFrame.pack(fill=tk.X)
        innerStartFrame.columnconfigure(0)
        innerStartFrame.columnconfigure(1, weight=1)
        
        radio_options_frame = tk.Frame(master=innerStartFrame, relief='flat', borderwidth=2, padx=5)
        radio_options_frame.grid(row = 0, column = 1, sticky='ew', padx=5)
        
        v = tk.IntVar()
        v.set(0)  # initializing the choice, i.e. Python

        radio_options = [("Yes", 0), ("No", 1)]

        def ShowChoice():
            MeasurerInstance.threshold = float(100) if v.get() == 0 else float(200)

        label = tk.Label(innerStartFrame, text="Include shadows?", pady=3)
        label.grid(row = 0, column = 0, sticky='w', padx=5)

        for option, val in radio_options:
            tk.Radiobutton(radio_options_frame, 
                        text=option,
                        padx = 5, 
                        variable=v, 
                        command=ShowChoice,
                        value=val).grid(row = 0, column = val, sticky='ew', padx=5)  
        
        ConstructApp.numberFramesSetting = DigitEntry("Number of Frames: ", 3, 1, innerStartFrame, trace="numberFrames", lower=0)
    
        self.startButton = tk.Button(startFrame, text='START', command=self.StartButton, bg="grey50", font=("Courier", 24), fg="white", state="disabled", disabledforeground="white")
        self.startButton.pack(fill=tk.BOTH, pady=5)
        
        paneeli_image.after(15, UpdateFeed) 
        
        # Loop to ensure calibration before training background
        def CheckIfCalibrated():
            if Cameras.GetSlope() is not None and Cameras.GetIntercept() is not None:
                self.backgroundButton["text"] = "TRAIN"
                self.backgroundButton["state"] = "normal"
                self.backgroundButton.configure(bg = "#74B224")
            else:
                self.backgroundButton["text"] = "PLS CALIBRATE"
                self.backgroundButton["state"] = "disabled"
                self.backgroundButton.configure(bg = "grey50")
                self.backgroundButton.after(500, CheckIfCalibrated)
            
        self.backgroundButton.after(15, CheckIfCalibrated)
        
        def on_closing():
            sys.stdout.close()
            rootWindow.destroy()

        rootWindow.protocol("WM_DELETE_WINDOW", on_closing)
        rootWindow.mainloop()
    
    def ButtonTextCountDown(self, label, thread):
        self.backgroundButton["text"] = str(label)

        if not thread.is_alive():
            self.backgroundButton["text"] = "RESTART"
            self.backgroundButton["state"] = "normal"
            self.backgroundButton.configure(bg = "#185CA8")
            
            self.startButton["state"] = "normal"
            self.startButton.configure(bg = "#74B224")
        else:
            if (isinstance(label, int) or isinstance(label, float)) and label != 0:
                self.backgroundButton.after(1000, self.ButtonTextCountDown, label-1, thread)
            else:
                self.backgroundButton.after(1000, self.ButtonTextCountDown, "...", thread)
                
    def BackgroundButtonsActive(self, status):
        if status:
            self.backgroundButton["state"] = "normal"
            self.cam_choice_dropdown["state"] = "normal"
            for child in self.inputsFrame.winfo_children():
                child.configure(state='normal')
        else:
            self.backgroundButton["state"] = "disabled"
            self.cam_choice_dropdown["state"] = "disabled"
            for child in self.inputsFrame.winfo_children():
                child.configure(state='disable')
            
    def BackgroundButton(self):
        if self.backgroundButton["text"] == "TRAIN":
            self.BackgroundButtonsActive(False)
            
            self.measurer = MeasurerInstance(ConstructApp.folder_path.get(), ConstructApp.outputFormatVariable.get())
            
            def ErrorCheckLoop():
                state, message = MeasurerInstance.error
                if (state):
                    tk.messagebox.showerror("Error", message)
                    MeasurerInstance.error = (False, "")
                    self.backgroundButton.after(10000, ErrorCheckLoop)
                    
                else:
                    self.backgroundButton.after(1000, ErrorCheckLoop)
                    
            self.backgroundButton.after(15, ErrorCheckLoop)
                    
            # Thread the Tkinter button countdown
            # x = threading.Thread(target=ConstructApp.ButtonTextCountDown, args=(self.backgroundButton, 10,), daemon=True)
            # x.start()
            x = threading.Thread(target=self.measurer.TrainBackground, daemon=True)
            x.start()
            
            self.backgroundButton.after(1000, self.ButtonTextCountDown, 20, x)
            
        elif self.backgroundButton["text"] == "RESTART":
            # Reconfigure buttons and delete the previous instance
            self.backgroundButton["text"] = "TRAIN"
            self.backgroundButton.configure(bg = "#74B224")
            self.startButton["state"] = "disabled"
            self.startButton.configure(bg = "grey50")
            
            # Unlock settings
            self.BackgroundButtonsActive(True)
            
            Cameras.DisconnectMeasurer()
            self.measurer = None
            
    def StartButton(self):
        self.backgroundButton["state"] = "disabled"
        self.startButton["state"] = "disabled"
        # ConstructApp.thresholdSetting.Activate(False)
        ConstructApp.numberFramesSetting.Activate(False)
        ConstructApp.fishIDEntry["state"] = "disabled"
        ConstructApp.additionalText["state"] = "disabled"
        ConstructApp.w["state"] = "disabled"
        ConstructApp.button["state"] = "disabled"
        
        self.measurer.outputFolder = ConstructApp.folder_path.get()
        self.measurer.format = ConstructApp.outputFormatVariable.get()
        
        skeletonThread = threading.Thread(target=Cameras.TriggerAnalysis, daemon=True)
        skeletonThread.start()
    
        self.startButton.after(1000, self.ReinstateSetting, skeletonThread)
    
    def ReinstateSetting(self, thread):
        if thread.is_alive():
            if MeasurerInstance.processingFrame is None:
                self.startButton["text"] = "COLLECTING..."
            else:
                self.startButton["text"] = "FRAME " + str(MeasurerInstance.processingFrame + 1) + "/" + str(int(Cameras.number_of_frames))
            
            self.startButton.after(1000, self.ReinstateSetting, thread)
        else:
            # ConstructApp.thresholdSetting.Activate(True)
            ConstructApp.numberFramesSetting.Activate(True)
            self.startButton["state"] = "normal"
            self.startButton["text"] = "START"
            self.backgroundButton["state"] = "normal"
            ConstructApp.fishIDEntry["state"] = "normal"
            ConstructApp.additionalText["state"] = "normal"
            ConstructApp.w["state"] = "normal"
            ConstructApp.button["state"] = "normal"
            

            
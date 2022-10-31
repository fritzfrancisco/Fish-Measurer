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

class TkinterApp():
    def __init__(self, **kwargs):
        
        # Error handling
        self.block_start_already_popped = False
        self.interrupt_already_popped = False
        
        # The state of execution of the app
            ## 0: base state
            ## 1: background is trained
            ## 2: running analysis
        self.current_state = 0
        
        # Principal components
        self.startButton = None
        self.backgroundButton = None
        
        ## consider removing all unnecessary class variables
        
        
        
        
        
        
        
        
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
            fishID = TkinterApp.fishIDEntry.get()
            addText = TkinterApp.additionalText.get("1.0",'end-1c')
            
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
        TkinterApp.exposureSetting = DigitEntry("Exposure (ms): ", 100, 0, self.inputsFrame, trace="exposure")
        
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
        TkinterApp.gainVariable = StringVar()
        TkinterApp.gainVariable.set(GAINOPTIONS[0]) # default value
        SendGain(GAINOPTIONS[0])
        w = OptionMenu(self.inputsFrame, TkinterApp.gainVariable, *GAINOPTIONS, command=SendGain)
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
        TkinterApp.wb_variable = StringVar()
        TkinterApp.wb_variable.set(WBOPTIONS[2]) # default value
        SendWB(WBOPTIONS[2])
        w = OptionMenu(self.inputsFrame, TkinterApp.wb_variable, *WBOPTIONS, command=SendWB)
        w["highlightthickness"] = 0
        w.grid(row = 2, column = 1, sticky='ew', padx=5)
        white_balance_setting.grid(row = 2, column = 0, sticky='w', padx=5)
        
        TkinterApp.frameRateSetting = DigitEntry("Framerate (fps): ", 30, 3, self.inputsFrame, trace="framerate")

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
                TkinterApp.folder_path.set(folderName)
                print(folderName)
            
        TkinterApp.button = tk.Button(browseFrame, text='Browse...', command=BrowseButton)
        TkinterApp.button.grid(row=0, column=0, sticky="nsew")

        TkinterApp.folder_path = StringVar()
        TkinterApp.folder_path.set(os.getcwd())
        selectedFolder = Label(master=browseFrame, textvariable=TkinterApp.folder_path, width=31, anchor='e')
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
        TkinterApp.outputFormatVariable = StringVar()
        TkinterApp.outputFormatVariable.set(OPTIONS[0]) # default value
        TkinterApp.w = OptionMenu(formatFrame, TkinterApp.outputFormatVariable, *OPTIONS)
        TkinterApp.w["highlightthickness"] = 0
        TkinterApp.w.grid(row = 0, column = 1, sticky='ew', padx=5, pady=5)

        # Fish ID and freetext
        watermarkFrame = tk.Frame(master=outputSettingsFrame, relief='flat', borderwidth=2, bg="grey60", padx=5, pady=2)
        watermarkFrame.pack(fill=tk.X)

        ## Fish ID
        fishIDFrame = tk.Frame(master=watermarkFrame, relief='flat', borderwidth=2)
        fishIDFrame.columnconfigure(0)
        fishIDFrame.columnconfigure(1, weight=1)
        fishIDFrame.pack(fill=tk.X, pady=3)

        fishID = tk.Label(text="Fish ID: ", master=fishIDFrame, pady=3)
        TkinterApp.fishIDEntry = tk.Entry(fishIDFrame, justify='left') 
        fishID.grid(row = 0, column = 0, sticky='w', padx=5)
        TkinterApp.fishIDEntry.grid(row = 0, column = 1, sticky='ew', padx=5)

        ## Freetext
        freetextFrame = tk.Frame(master=watermarkFrame, relief='flat', borderwidth=2, padx=5)
        freetextFrame.pack(fill=tk.BOTH, pady=4)

        freetextPrompt = tk.Label(text="Additional text for watermark:", master=freetextFrame, anchor="w")
        freetextPrompt.config(font=("Courier", 10))
        freetextPrompt.pack(fill=tk.X)

        TkinterApp.additionalText = tk.Text(freetextFrame, height=5, width=5)
        TkinterApp.additionalText.pack(fill=tk.X, pady=5)
        
        ## -- BG BUTTON -----------------------------------------------------------------------------------
        backgroundFrame = tk.Frame(master=settingsFrame, relief='flat', borderwidth=2, padx=10, pady=10, bg="grey80")
        backgroundFrame.pack(fill=tk.BOTH)
        
        self.backgroundButton = tk.Button(backgroundFrame, text='TRAIN', command=self.BackgroundButtonClick, bg="#74B224", font=("Courier", 24), fg="white", disabledforeground="white")
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
        
        TkinterApp.numberFramesSetting = DigitEntry("Number of Frames: ", 3, 1, innerStartFrame, trace="numberFrames", lower=0)
    
        self.startButton = tk.Button(startFrame, text='START', command=self.StartButtonClick, bg="grey50", font=("Courier", 24), fg="white", state="disabled", disabledforeground="white")
        self.startButton.pack(fill=tk.BOTH, pady=5)
        
        paneeli_image.after(15, UpdateFeed) 
        
        def on_closing():
            sys.stdout.close()
            rootWindow.destroy()

        rootWindow.protocol("WM_DELETE_WINDOW", on_closing)
        rootWindow.mainloop()
        
        
        
        
        
        
        
    def CheckIfCalibrated(self):
        # Loop to ensure calibration before training background
        if Cameras.GetSlope() is not None and Cameras.GetIntercept() is not None:
            self.ActivateBackgroundButton(True)
        else:
            self.backgroundButton["text"] = "PLS CALIBRATE"
            self.ActivateBackgroundButton(False)
            self.backgroundButton.after(500, self.CheckIfCalibrated)
        
    def BackgroundButtonProcessing(self, label, thread):
        self.backgroundButton["text"] = label

        if not thread.is_alive():
            self.current_state = 1
            self.ActivateBackgroundButton(True)
            self.ActivateStartButton(True)
        
        if not self.measurer_instance.pulling_background and label[0] != "P":
            self.backgroundButton.after(1000, self.BackgroundButtonProcessing, "Processing", thread)
        else:
            if label[-3:] == "...":
                self.backgroundButton.after(1000, self.BackgroundButtonProcessing, label[:-3], thread)
            else:
                self.backgroundButton.after(1000, self.BackgroundButtonProcessing, label + ".", thread)
                
    def StartCheckingForErrors(self):
        # INTERRUPTION ERRORS
        ## There will only ever be one at a time, as interruption errors end the analysis and 
        ## therefore architecturally preclude the raising of additional interruption errors.
        ## App should resume normal functionality as the analysis thread will close cleanly
        if self.measurer_instance.errors["interrupt"]:
            # Don't raise the pop-up if it's already been raised for this event
            if not self.interrupt_already_popped:
                self.interrupt_already_popped = True
                tk.messagebox.showerror("Analysis Error", self.measurer_instance.errors["interrupt"][0])
                self.measurer_instance.errors["interrupt"] = []
        else:
            self.interrupt_already_popped = False
        
        # START-BLOCK ERRORS
        if self.measurer_instance.block_tkinter_start_button:
            self.ActivateStartButton(False)
            
            # Don't raise the pop-up if it's already been raised for this event
            if not self.block_start_already_popped:
                self.block_start_already_popped = True
                tk.messagebox.showerror("Shape is Missing!", "Failing to register any objects in the arena (quite the feat). Please ensure an object is present and contrasted against the trained background")
        else:
            self.ActivateStartButton(True)
            self.block_start_already_popped = False
            
        self.backgroundButton.after(100, self.StartCheckingForErrors)
    
    def ActivateStartButton(self, activate):
        if activate:
            if self.current_state == 0:
                self.startButton.configure(bg="grey50", state="disabled")
            elif self.current_state == 1:
                self.startButton.configure(bg="#74B224", state="normal")
            elif self.current_state == 2:
                self.startButton.configure(bg="grey50", state="disabled")
        else:
            self.startButton.configure(bg="grey50", state="disabled")
    
    def ActivateBackgroundButton(self, activate):
        if activate:
            if self.current_state == 0:
                self.backgroundButton.configure(text='TRAIN', bg="#74B224", state="normal")
            elif self.current_state == 1:
                self.backgroundButton.configure(text='RESTART', bg ="#185CA8", state="normal")
            elif self.current_state == 2:
                self.backgroundButton.configure(bg="grey50", state="disabled")
        else:
            self.backgroundButton.configure(bg="grey50", state="disabled")
            
    def LockSettings(self, lock):
        if not lock:
            self.ActivateBackgroundButton(True)
            self.cam_choice_dropdown["state"] = "normal"
            for child in self.inputsFrame.winfo_children():
                child.configure(state='normal')
        else:
            self.ActivateBackgroundButton(False)
            self.cam_choice_dropdown["state"] = "disabled"
            for child in self.inputsFrame.winfo_children():
                child.configure(state='disable')
            
    def BackgroundButtonClick(self):
        if self.current_state == 0:
            self.LockSettings(True)
            self.measurer_instance = MeasurerInstance()
                    
            # Thread the Tkinter button countdown
            x = threading.Thread(target=self.measurer_instance.TrainBackground, daemon=True)
            x.start()
            
            self.backgroundButton.after(1000, self.BackgroundButtonProcessing, "Gathering", x)
            
        elif self.current_state == 1 or self.current_state == 2:
            self.current_state = 0
            
            # Reconfigure buttons and delete the previous instance
            # LockSettings() passively handles backgroundButton activation
            self.ActivateStartButton(True)
            self.LockSettings(False)
            
            Cameras.DisconnectMeasurer()
            self.measurer_instance = None
            
    def StartButtonClick(self):
        self.current_state = 2
        
        self.ActivateBackgroundButton(False)
        self.ActivateStartButton(False)
        
        TkinterApp.numberFramesSetting.Activate(False)
        TkinterApp.fishIDEntry["state"] = "disabled"
        TkinterApp.additionalText["state"] = "disabled"
        TkinterApp.w["state"] = "disabled"
        TkinterApp.button["state"] = "disabled"
        
        # These are now locked
        self.measurer_instance.outputFolder = TkinterApp.folder_path.get()
        self.measurer_instance.format = TkinterApp.outputFormatVariable.get()
        
        skeletonThread = threading.Thread(target=Cameras.TriggerAnalysis, daemon=True)
        skeletonThread.start()
    
        self.startButton.after(1000, self.ReinstateSettings, skeletonThread)
    
    def ReinstateSettings(self, thread):
        if thread.is_alive():
            if MeasurerInstance.processingFrame is None:
                self.startButton["text"] = "COLLECTING..."
            else:
                self.startButton["text"] = "FRAME " + str(MeasurerInstance.processingFrame + 1) + "/" + str(int(Cameras.number_of_frames))
            
            self.startButton.after(1000, self.ReinstateSettings, thread)
        else:
            self.current_state = 1
            
            print("reinstating settings")
            TkinterApp.numberFramesSetting.Activate(True)
            self.ActivateBackgroundButton(True)
            self.ActivateStartButton(True)
            
            TkinterApp.fishIDEntry["state"] = "normal"
            TkinterApp.additionalText["state"] = "normal"
            TkinterApp.w["state"] = "normal"
            TkinterApp.button["state"] = "normal"
            

            
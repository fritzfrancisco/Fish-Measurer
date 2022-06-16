import tkinter as tk
from tkinter import *
from MeasurerInstance import MeasurerInstance
from Cameras import Cameras

class DigitEntry():
    
    '''A Entry widget that only accepts digits'''
    def __init__(self, labelText, defaultValue, row, master, **kwargs):
        
        self.lower = kwargs["lower"] if "lower" in kwargs else None
        self.upper = kwargs["upper"] if "upper" in kwargs else None
        self.trace = kwargs["trace"] if "trace" in kwargs else None
        self.master = master
        
        if "pack" in kwargs:
            # frame = tk.Frame(master=self.master, relief='flat', borderwidth=2, pady=5)
            self.master.pack(fill=tk.X)
            self.master.columnconfigure(0)
            self.master.columnconfigure(1, weight=1)
        
        self.label = tk.Label(text=labelText, master=self.master, pady=3)
        
        self.var = tk.StringVar(self.master)
        self.var.trace('w', self.validate)
        self.get, self.set = self.var.get, self.var.set
        self.set(defaultValue)
        
        self.entry = tk.Entry(self.master, textvariable=self.var, justify='center') 
        self.label.grid(row = row, column = 0, sticky='w', padx=5)
        self.entry.grid(row = row, column = 1, sticky='ew', padx=5)
        
    def validate(self, *args):
        value = self.get()
        if not value.isdigit():
            self.set(''.join(x for x in value if x.isdigit()))
            value = self.get()
        
        if self.lower:
            if float(value) < self.lower:
                self.set(self.lower)
        if self.upper:
            if float(value) > self.upper:
                self.set(self.upper)
        if not (not self.trace):
            if self.trace == "exposure":
                Cameras.currentCam.ExposureTime.SetValue(float(self.get()))
            elif self.trace == "framerate":
                Cameras.currentCam.AcquisitionFrameRateEnable.SetValue(True)
                Cameras.currentCam.AcquisitionFrameRate.SetValue(float(self.get()))
            else:
                MeasurerInstance.threshold = float(self.get())
    
    def ChangeValue(self, newVal):
        self.set(newVal)
        
    def GetValue(self):
        return self.get()
    
    def Activate(self, status):
        if status:
            self.entry["state"] == "disabled"
        else:
            self.entry["state"] == "normal"
            
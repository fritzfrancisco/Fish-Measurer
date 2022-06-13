import tkinter as tk
from tkinter import *
import Cameras
import MeasurerInstance

class DigitEntry():
    
    '''A Entry widget that only accepts digits'''
    def __init__(self, labelText, defaultValue, row, master=None, **kwargs):
        
        self.lower = kwargs["lower"] if "lower" in kwargs else None
        self.upper = kwargs["upper"] if "upper" in kwargs else None
        self.trace = kwargs["trace"] if "trace" in kwargs else None
        
        self.label = tk.Label(text=labelText, master=master, pady=3)
        
        self.var = tk.StringVar(master)
        self.var.trace('w', self.validate)
        self.get, self.set = self.var.get, self.var.set
        self.set(defaultValue)
        
        self.entry = tk.Entry(master, textvariable=self.var, justify='center') 
        self.label.grid(row = row, column = 0, sticky='w', padx=5)
        self.entry.grid(row = row, column = 1, sticky='ew', padx=5)
        
    def validate(self, *args):
        value = self.get()
        if not value.isdigit():
            self.set(''.join(x for x in value if x.isdigit()))
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
            self.config(state="normal")
        else:
            self.config(state="disabled")
            
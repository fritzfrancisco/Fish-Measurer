from Cameras import Cameras

class MeasurerInstance():
    
    def __init__(self, settingsDict):
        self.settings = settingsDict["settings"] 
        self.watermark = settingsDict["watermark"]
        self.outputFolder = settingsDict["folder"]
        self.format = settingsDict["format"]
        
        Cameras.ApplySettings(exposure=self.settings["exposure"], gain=self.settings["gain"],
                              framerate=self.settings["framerate"], duration=self.settings["duration"])

        
        
        
        #camera.StartGrabbingMax(images_to_grab, pylon.GrabStrategy_OneByOne)
        
        
        
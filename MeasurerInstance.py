

class MeasurerInstance():
    
    def __init__(self, settingsDict):
        self.settings = settingsDict["settings"] 
        self.watermark = settingsDict["watermark"]
        self.outputFolder = settingsDict["folder"]
        self.format = settingsDict["format"]
        
        
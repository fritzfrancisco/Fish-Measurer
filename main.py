from Cameras import Cameras
from app_constructor import TkinterApp


# For debug logs --> uncomment to specify a location for these logs
# import sys
# import os
# from datetime import datetime
# desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
# folder_path = os.path.join(desktop, "FishMeasurerDebugLogs")
# if not os.path.isdir(folder_path):
#     os.mkdir(folder_path)
# log_string = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) + ".txt"
# full_path = os.path.join(folder_path, log_string)
# sys.stdout = open(full_path, 'w')
# sys.stderr = sys.stdout

# Actual execution
camera = Cameras()
app = TkinterApp()
app.CheckIfCalibrated()
app.Run()


# PyInstaller
# Open Anaconda prompt and activate the cam-measurer environment: activate cam-measurer
# Navigate to directory: cd C:\Users\james\Desktop\04_Cam-Measurer\01_Code\02_Spec-Files
# Run it: pyinstaller main.spec


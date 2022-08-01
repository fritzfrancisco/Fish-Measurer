from Cameras import Cameras
from app_constructor import ConstructApp
import sys
import os
from datetime import datetime

# # For debug logs
# desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
# folder_path = os.path.join(desktop, "FishMeasurerDebugLogs")
# if not os.path.isdir(folder_path):
#     os.mkdir(folder_path)
    
# log_string = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")) + ".txt"
# full_path = os.path.join(folder_path, log_string)
# sys.stdout = open(full_path, 'w')
# sys.stderr = sys.stdout

# Actual execution
camera = Cameras()
app = ConstructApp()


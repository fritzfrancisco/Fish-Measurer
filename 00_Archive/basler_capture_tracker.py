#!/usr/bin/env python3

import serial
import os
import json
import numpy as np
import cv2
import argparse
from datetime import datetime, time
from pypylon import pylon

ap = argparse.ArgumentParser(
    description='Input specific output pathway and framerate')
ap.add_argument("-o", "--output_directory", required=False,
                default='', help="path to output directory")
ap.add_argument("-wo", "--width", required=False,
                default=2048, type=int, help="output width")
ap.add_argument("-ho", "--height", required=False,
                default=2048, type=int, help="output height")
ap.add_argument("-r", "--framerate", type=int, default=30,
                required=False, help="acquisition frame rate")
ap.add_argument("-c", "--chunk_size", required=False,
                default=50000, type=int, help="output chunk size in frames")
ap.add_argument("-sc", "--scaling", required=False, default=1,
                type=float, help="scaling factor for output e.g 1, 0.5, 0.25")
ap.add_argument("-cc", "--codec", required=False, default='mp4v',
                type=str, help="video codec [FOURCC]")
ap.add_argument("-exp", "--exposure", required=False,
                default=50000, type=int, help="exposure time")
ap.add_argument("-gs", "--gain_setting", required=False,
                default='Continuous', type=str, help="gain settings ['Once', 'Continuous', 'Off']")
ap.add_argument("-s", "--show", default=True, help="show live video")
ap.add_argument("-d", "--duration", default=-1, type=int,
                help="video duration (seconds)")
ap.add_argument("-f", "--filetype", default='.mp4', type=str,
                help="video filetype extension")
ap.add_argument("-t", "--tracking", default=False, type=bool,
                help="flag for closed loop tracking")
ap.add_argument("-min", "--min_area", default=1, type=int,
                help="minimum  contour area in px")
ap.add_argument("-max", "--max_area", default=1000, type=int,
                help="maximum contour area in px")
ap.add_argument("-g", "--grid", default=[None,None], type=json.loads,
                help="NxN grid to separate field of view by")
ap.add_argument("-ser", "--serial", default=False, type=bool,
                help="Flag setting serial communication or not")
ap.add_argument("-th", "--threshold", default=[0,255], type=json.loads,
                help="min,max of threshold to be applied to image")
ap.add_argument("-cr", "--crop", default=None, type=json.loads,
                help="apply circular cropping with radius [radius,center_x,center_y] to image")
ap.add_argument("-n", "--number", default=0, type=int,
                help="Maximum number of individuals expected to be tracked")
args = vars(ap.parse_args())
print(args)

def open_serial(com='/dev/ttyUSB0', baudrate=115200):
    '''Create serial connection to com'''
    ser = serial.Serial()
    ser.baudrate = baudrate
    ser.port = com
    ser.open()
    print(str("Serial connection established @ " + com + " with baudrate:" + str(baudrate)))
    return ser

def check_center_coordinates(centers,grid):
    '''function to find which grid cell a coordinate point is in
    and summarizing occurrences for each cell'''
    counts = np.zeros(grid.shape[0])
    for i, arena in enumerate(grid):
        counts[i] = np.sum([1 if (c[0] < arena[2]) and (c[0] > arena[0]) and (c[1] < arena[3]) and (c[1] > arena[1]) else 0 for c in centers])
    return counts

def plot_grid(img):
    '''plot grid onto image'''
    for col in np.arange(np.array(args['grid'])[0]):
        img = cv2.line(img, (int(col * args['width']/np.array(args['grid'])[0]),0), (int(col * args['width']/np.array(args['grid'])[0]),args['height']), (255,0,0), 5)
    for row in np.arange(np.array(args['grid'])[1]):
        img = cv2.line(img, (0,int(row * args['height']/np.array(args['grid'])[1])), (args['width'], int(row * args['height']/np.array(args['grid'])[1])), (255,0,0), 5)
    return img

def change_chunk():
    '''function that changes chunk number and output file accordingly.
    It accesses the global input arguments as well as chunk_idx to do so.'''
    global camera, args, chunk_idx
    dateTimeObj = datetime.now()
    timestamp = dateTimeObj.strftime("%Y%m%d_%H%M%S")
    fourcc = cv2.VideoWriter_fourcc(*args['codec'])
    fname = str(str(args['output_directory']) + str(chunk_idx).zfill(6) + '_' + str(camera.GetDeviceInfo().GetModelName()).replace('-', '_') + '_' + str(camera.GetDeviceInfo().GetSerialNumber()) + '_exp' + str(args['exposure']) +
                '_r' + str(args["framerate"]) + '_res' + str(args['scaling']) + '_' + str(timestamp) + str(args['filetype']))
    out = cv2.VideoWriter(filename=fname, fourcc=fourcc,
                          fps=25, frameSize=(int(args['width']*args['scaling']),int(args['height']*args['scaling'])),isColor=1)
    ret = False
    print('Output file: ', fname)
    return ret, out


# Create timestamp
dateTimeObj = datetime.now()
timestamp = dateTimeObj.strftime("%Y%m%d_%H%M%S")

# Conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
print('Camera: ', camera)

# Open camera instance
camera.Open()

# Set exposure time
camera.ExposureTime.SetValue(args['exposure'])

# Set gain
camera.GainAuto.SetValue(args['gain_setting'])

# Set acquisition frame rate
camera.AcquisitionFrameRateEnable.SetValue(True)
camera.AcquisitionFrameRate.SetValue(float(args['framerate']))
print('Image Aquisition Rate :', camera.AcquisitionFrameRate.GetValue())

# Set image dimensions
camera.Width.SetValue(args['width'])
camera.Height.SetValue(args['height'])

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
# camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
images_to_grab = args['framerate'] * args['duration']
#camera.StartGrabbingMax(images_to_grab, pylon.GrabStrategy_OneByOne)


# Set up video window
if args['show'] == True:
    cv2.namedWindow(str('Basler Capture ' +
                    str(camera.GetDeviceInfo().GetSerialNumber())), cv2.WINDOW_NORMAL)

# converting to opencv bgr format
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# frame counter
frame_idx = 0

# chunk counter
chunk_idx = 0

# initate with first chunk
change, out = change_chunk()

# initiate serial connection
if args['serial'] == True:
    ser = open_serial(com='/dev/ttyUSB0', baudrate=115200)

# Background subtraction
backSub = cv2.createBackgroundSubtractorMOG2(history=100)
backSub.setShadowValue(0)

# iniated tracking variables
if (args['tracking'] != False) and (np.array(args['grid']).all()!= None):
    grid = []
    for col in np.arange(np.array(args['grid'])[0]):
        for row in np.arange(np.array(args['grid'])[1]):
            xmin = col * args['width']/int(args['grid'][0])
            xmax = (col+1) * args['width']/int(args['grid'][0])
            ymin = row * args['height']/int(args['grid'][1])
            ymax = (row+1) * args['height']/int(args['grid'][1])
            grid.append([xmin,ymin,xmax,ymax])
    grid = np.array(grid).reshape(-1,4)
    print('Selection grid:')
    print(grid)

# initiate cropping variables
if args['crop'] != None:
    r = np.array(args['crop'])[0]
    cx = np.array(args['crop'])[1]
    cy = np.array(args['crop'])[2]

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():

        # Change chunk
        if change == True:
            change, out = change_chunk()

        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        if args['scaling'] != 1:
            # Resize output image based on scaling
            img = cv2.resize(img, None, fy=args['scaling'], fx=args['scaling'])

        # show circular cropping
        if args['crop'] != None:
            mask = np.zeros((int(args['width']),int(args['height']),3),dtype=np.uint8)
            cv2.circle(mask,(cx,cy),r,(255,255,255),-1,8,0)
            out_img = img*mask
            white = 255-mask
            img = out_img + white

        if args['tracking'] != False:
            # Simple thresholding and object detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = backSub.apply(gray)
            #_,img = cv2.threshold(gray,np.array(args['threshold'])[0],np.array(args['threshold'])[1],cv2.THRESH_BINARY)
            img = cv2.GaussianBlur(img,(5,5),0) # was 13
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (11,11)) # was 11
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, (11,11))
            img = cv2.dilate(img,(5,5),iterations = 1)
            contours,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_GRAY2BGR)
            
            selected_contours = []
            if len(contours) <= args['number']:
                for c in contours:
                    contour_area = cv2.contourArea(c)
                    x,y,w,h = cv2.boundingRect(c)        
                    bounding_rect_area = w*h
                    if(contour_area > args['min_area'] and contour_area < args['max_area']):
                        selected_contours.append(c)

                    cv2.drawContours(img, selected_contours, -1, (0,0,255), thickness=cv2.FILLED)
                ret = True

            else:
                occurrences = [0,0]
                ret = False

            if np.array(args['grid']).all() != None:
                centers = np.array([(cv2.moments(c)["m10"]/cv2.moments(c)["m00"],cv2.moments(c)["m01"]/cv2.moments(c)["m00"]) for c in selected_contours])
                occurrences = check_center_coordinates(centers,grid)
                # print(occurrences)
                img = plot_grid(img)
            
            img = cv2.putText(img, str("Found " + str(len(selected_contours)).zfill(4) + " contours"), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0), 8)

        # show cropping circle
        if args['crop'] != None:
            img = cv2.circle(img,(int(cx),int(cy)), int(r), (255,0,0), 4, lineType=cv2.LINE_AA)

        # add timestamp in output image
        img = cv2.putText(img, datetime.now().strftime("%Y%m%d %H:%M:%S"), (
            15, img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        

        if (args['serial'] == True) and (args['tracking'] == True):
            # send information to ESP
            # send_value = round(int(occurrences[0])/10)*10
            # normalize occurences to fit withing RGB range of 0-255
            occurrences = (int((255/int(args['number']))) * occurrences)
            print(occurrences)
            send_value_1 = int(occurrences[0])
            send_value_2 = int(occurrences[-1])
            values = bytearray([send_value_1, send_value_2])
            
            ser.write(values)

            # print sent values
            for i in np.arange(len(values)):
                print('value', i, ': ', ord(ser.read(1)))
            
        if args['output_directory'] != None:
      
            # Convert image to gray scale
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Save to video container
            out.write(img)

        if args['show'] == True:
            cv2.imshow(str('Basler Capture ' +
                       str(camera.GetDeviceInfo().GetSerialNumber())), img)
            k = cv2.waitKey(1)
            if k == 27:
                camera.StopGrabbing()
                camera.Close()
                if args['output_directory'] != None:
                    out.release()
                if args['show'] == True:
                    cv2.destroyAllWindows()
                break
        
        # End recording if duration is exceeded
        if (args['duration'] > 0) and (frame_idx >= int(args['duration']*args['framerate'])):
            camera.StopGrabbing()
            camera.Close()
            if args['output_directory'] != None:
                out.release()
            if args['show'] == True:
                cv2.destroyAllWindows()
            break

        # print(img.shape)

    else:
        if args['output_directory'] != None:
            
            # Create black image
            blank = np.ones((img.shape[0],img.shape[1],3),np.uint8) * 255

            # Save to video container
            out.write(blank)

        # End recording if duration is exceeded
        if (args['duration'] > 0) and (frame_idx >= int(args['duration']*args['framerate'])):
            camera.StopGrabbing()
            camera.Close()
            if args['output_directory'] != None:
                out.release()
            if args['show'] == True:
                cv2.destroyAllWindows()
            break
        print('Grab result unsuccessful!')
        # break

    grabResult.Release()
    frame_idx += 1

    # Change to next chunk
    if frame_idx % args['chunk_size'] == 0:
        change = True
        chunk_idx += 1

    # print(frame_idx)

# Releasing the resource
print('Finished recording! ',str(camera.GetDeviceInfo().GetSerialNumber()))
camera.StopGrabbing()
camera.Close()

if args['serial'] == True:
	ser.close()

if args['output_directory'] != None:
    out.release()
if args['show'] == True:
    cv2.destroyAllWindows()

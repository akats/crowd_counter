#!/usr/bin/python
# encoding: utf-8
import sys
import os
import time
import re
from os import path as osp

import cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import crowd
import draw
import crowdmisc

###############################################################################
## 'global' variables

newPoint = None

densityWindowSize = (480, 320)
stateWindowSize = (800, 300)
statisticWindowSize = (600, 400)
configImageZoom=0.5

sourceDir = "./images"
recycleDir = "./recycled"
statFileName = "./result.png"

# Yakimanka Feb 4
# pattern = re.compile("^im_\d{10}_\d{3}.jpg$")
# Jun 12, Rozhdestvensky Blvd
pattern = re.compile("^DSC_\d{4}.JPG$")

###############################################################################
## the mouse callback for 'Configure' window

def on_mouse(event, x, y, flags, param):
    global newPoint;
    if event == cv.CV_EVENT_LBUTTONDOWN:
        newPoint = (x, y)

camera = cv.CaptureFromCAM(-1)
if camera is None:
    print "Camera off"
    sys.exit(1)

def getFileTime(filename):
    return float(filename[4:8])# + float(filename[14:17])/1000

def getNextFrame(previousFrameTime):
    fileTime = 0
    image = None
    fileList = os.listdir(sourceDir)
    fileList.sort()

    # pattern now in global
    fileName = ""
    for fileName in fileList:
        if pattern.match(fileName):
            fileTime = getFileTime(fileName)
            if fileTime > previousFrameTime:
                filePath=osp.join(sourceDir, fileName)
                #print "File Time: %d", (fileTime)
                try:
                    image = cv.LoadImageM(filePath, cv.CV_LOAD_IMAGE_GRAYSCALE)
                except IOError: 
                    print "Can't read ", filePath
                    continue
                break
    return image, fileTime, len(fileList), fileName

#gray_frame = None
#def getNextFrame(previousFrameTime):
#    global gray_frame
#    frame = cv.QueryFrame(camera)
#    if gray_frame is None:
#        gray_frame = cv.CreateImage(cv.GetSize(frame), 8, 1)
#    cv.CvtColor(frame, gray_frame, cv.CV_BGR2GRAY)
#    return gray_frame, time.time(), 0


# Marks frames (handled) before beforeTime
def markFrames(beforeTime):
    fileList = os.listdir(sourceDir)
    fileList.sort()
    #pattern = re.compile("^im_\d{10}_\d{3}.jpg$")
    for fileName in fileList:
        if pattern.match(fileName):
            fileTime = getFileTime(fileName)
            if fileTime <= previousFrameTime:
                os.rename(osp.join(sourceDir, fileName),
                          osp.join(recycleDir + fileName))
            else:
                break
            
def drawStreetRegion(DestImage, StreetImage):
    #cv.SetImageROI(DestImage, (DestImage.width/4, 0, 3*DestImage.width/4, DestImage.height))
    cv.Resize(StreetImage, DestImage)
    #cv.ResetImageROI(DestImage)

def saveStatImage(image, time):
    try:
        saveFileName = "%s_%d.jpg" % (statFileName, int(time))
        print saveFileName
        cv.SaveImage('tempstat.jpg', image)
        os.rename('tempstat.jpg', saveFileName)
    except IOError: 
        print "Can't save ", saveFileName

def logState(filename, time, peopleDensity, currPeopleCount):
    try:
        logFile = open('file.log', 'a')
        s = "%s;%u;%0.3f;%.6f;\n" % (filename , time , peopleDensity , currPeopleCount)
        logFile.write(s)
        logFile.close()
    except IOError: 
        print "Cann't save to file.log"

def restoreState():
    firstFrameTime = 0.0
    lastFrameTime = 0.0
    currPeopleCount = 0.0
    try:
        logFile = open('file.log', 'r')
        lines = logFile.readlines();
        for line in lines:
            fields = line.split(';')
            if firstFrameTime <= 0.0:
                firstFrameTime = float(fields[1])
            lastFrameTime = float(fields[1])
            currPeopleCount = float(fields[3])
        logFile.close()
    except IOError: 
        print "Cann't restore a state from log.file"
    return firstFrameTime, lastFrameTime, currPeopleCount
    
def main(sourceDir_, recycleDir_, statFileName_, waitKey = cv.WaitKey):
    global sourceDir, recycleDir, statFileName
    sourceDir = sourceDir_
    recycleDir = recycleDir_
    statFileName = statFileName_
    global newPoint, previousFrameTime
        
    # Algorithm parameters
    sampleRectDims = crowdmisc.sample_rect_dims


    # Variables
    processMode = "pause"

    trackingRegions=None


    delayBetweenFrames = 500 # ms. For 'delay' mode
    gotoNextFrame = 0        # allow go to next frame in 'pause' mode

    crowdDensity = 0.
    numPeopleInSample = 0
    currPeopleCount = 0

    peoplePerStreetRegStr = ""

    currentFileName = ""

    rgbFrame = None

    currentFrame = None
    previousFrame = None

    firstFrameTime = 0
    currentFrameTime = 0
    previousFrameTime = 0
    frameQueueLenght = 0
    
    recomputeDensityFlag=True

    # Set up windows
    configWindow='flowCorners'
    configWindowImage = None

    densityWindow='Density'
    densityWindowImage = cv.CreateImage(densityWindowSize, 8, 3)

    stateWindow='State'
    stateWindowImage = cv.CreateImage(stateWindowSize, 8, 3)

    statisticWindow='Statistic'
    statisticWindowImage = cv.CreateImage(statisticWindowSize, 8, 3)

    cv.NamedWindow(configWindow)
    cv.SetMouseCallback(configWindow, on_mouse)
    
    cv.NamedWindow(densityWindow, 0)
    cv.NamedWindow(statisticWindow)
    cv.NamedWindow(stateWindow)

    #############################
    # Load saved state.  COMMENT TO RESET EVERY TIME
    #############################
    firstFrameTime, previousFrameTime, currPeopleCount = restoreState()

    # Take one first and second frames
    frame, previousFrameTime, frameQueueLenght, currentFileName = \
        getNextFrame(previousFrameTime)
    gotoNextFrame = 1
    if firstFrameTime <= 0.0:
        firstFrameTime = previousFrameTime

    trackingRegions = crowdmisc.constructTrackingRegions(frame)
    currTrackingRegions = crowdmisc.constructTrackingRegions(frame)

    # Main Loop
    delayTime = 1;
    while True:
        # Determine how long to pause to check for key input, depending on 
        # the state of the program.  If we check too often, we lose performance
        # but if we don't check often enough, the program becomes unresponsive.
        if processMode=='delay' :
            delayTime = delayBetweenFrames
        elif processMode in ('pause', 'egomotion') and gotoNextFrame == 0:
            delayTime = 250 # check for events four times a second
        elif processMode == 'pause' and gotoNextFrame > 0:
            delayTime = 1
        else:
            delayTime = 100
        
        key = waitKey(delayTime)
        if key >= 0:
            key = key & 0xff

            # Reset motion and egomotion tracking regions
            if key == ord('r'):
                trackingRegions = crowdmisc.constructTrackingRegions(currentFrame)

            # To edit number of people
            if key >= ord('0') and key <= ord('9'):
                peoplePerStreetRegStr = peoplePerStreetRegStr + chr(key)
            if key == 8 and peoplePerStreetRegStr != "":
                peoplePerStreetRegStr = peoplePerStreetRegStr[:-1]

            # 'Enter' pressed - save new number of people
            if key == 10 and peoplePerStreetRegStr != "":
                crowdDensity = crowd.calculate_density(
                    int(peoplePerStreetRegStr), sampleRectDims)
                numPeopleInSample = int(peoplePerStreetRegStr)
                peoplePerStreetRegStr = ""
                recomputeDensityFlag=True
                
            # Select mode
            if key == ord('p'):
                    processMode = "pause"
            if key == ord('e') and crowdmisc.egomotion_correction == True:
                    processMode = "egomotion"
            if key == ord('a') and not trackingRegions['flowWarpMatrix'] is None:
                    processMode = "auto"
            if key == ord('d') and not trackingRegions['flowWarpMatrix'] is None:
                    processMode = "delay"

            if processMode == "delay":
                if key == 82:    # UP
                    delayBetweenFrames += 1000
                elif key == 84:    # DOWN
                    delayBetweenFrames -= 1000
                elif key == 81:    # LEFT
                    delayBetweenFrames -= 100
                elif key == 83:    # RIGHT
                    delayBetweenFrames += 100

                if delayBetweenFrames < 1:
                    delayBetweenFrames = 1

            if processMode in ("pause", "egomotion"):
                if key == 82:    # UP
                    gotoNextFrame = 10
                if key == 83:    # RIGHT
                    gotoNextFrame = 1
                    
            if processMode == 'delay':
                delayTime = delayBetweenFrames

        # Quit from main loop
        if key == ord('q'):
            break

        # Check new corner point
        if not newPoint is None:
            if processMode == 'egomotion':
                crowdmisc.updateEgomotionTrackingRegion(
                    [newPoint], configImageZoom, trackingRegions)
            else:
                crowdmisc.updateFlowTrackingRegion(
                    [newPoint], configImageZoom, trackingRegions)
            newPoint = None
            currTrackingRegions = trackingRegions

        ##################################
        # Get next frame
        if not processMode in ("pause", "egomotion") or gotoNextFrame > 0:
            frame, tm, frameQueueLenght, currentFileName = getNextFrame(previousFrameTime)
            if not frame is None:
                currentFrameTime = tm
            gotoNextFrame -= 1

        if not frame is None:
            if firstFrameTime <= 0.0:
                firstFrameTime = currentFrameTime

            if rgbFrame is None:
                rgbFrame = cv.CreateImage(cv.GetSize(frame), 8, 3)
    
            if currentFrame is None:
                currentFrame = cv.CreateMat(frame.rows, frame.cols, frame.type)
            if previousFrame is None:
                previousFrame = cv.CreateMat(frame.rows, frame.cols, frame.type)
    
            if frame.channels == 1:
                cv.CvtColor(frame, rgbFrame, cv.CV_GRAY2BGR)
            else:
                cv.Copy(frame, rgbFrame)
    
            cv.Copy(frame, currentFrame)
    
            ##################################
            # Draw corner configuration
            if configWindowImage is None:
                configWindowImage = cv.CreateMat(int(frame.rows*configImageZoom),
                                                 int(frame.cols*configImageZoom),
                                                 cv.CV_8UC3)
    
            cv.Resize(rgbFrame, configWindowImage)

        #end if (not frame is None)

        draw.drawPoints(configWindowImage, currTrackingRegions['displayFlowCorners'])
        draw.drawPoints(configWindowImage,
                        currTrackingRegions['displayStableCorners'],
                        color = cv.RGB(0, 0, 255))

        ##################################
        # Prepare images for windows
        #cv.SetZero(densityWindowImage)
        cv.SetZero(stateWindowImage)
        # cv.SetZero(statisticWindowImage)

        ###################################
        # calculate result
        if not trackingRegions['flowWarpMatrix'] is None:
            # draw part of street
            if recomputeDensityFlag:
                densityWindowImage, sampleRegionBounds = draw.draw_sample_region( \
                    currentFrame, currTrackingRegions['flowWarpMatrix'], sampleRectDims)
                
                recomputeDensityFlag=False
                
                #print "redisplaying density image..."
            if currentFrameTime > previousFrameTime:
                # analyzing
                currPeopleCount, prevFeatures, currFeatures, \
                meanVelocity, velocityInlierIdx, currTrackingRegions = \
                    crowd.compute_pedestrian_flow(
                        previousFrame, currentFrame,
                        trackingRegions,
                        crowdDensity, currPeopleCount)
                # draw statistic to 
                statImage = draw.drawResult(
                    currentFrame,
                    prevFeatures, currFeatures,
                    meanVelocity, velocityInlierIdx,
                    crowdDensity, numPeopleInSample,
                    currPeopleCount,
                    currentFrameTime, sampleRegionBounds, currTrackingRegions)
                cv.Resize(statImage, statisticWindowImage)
                saveStatImage(statImage, currentFrameTime)
                sampleRegionBounds=None

        ##################################
        ## Draw text
        if not configWindowImage is None:
            draw.drawText(configWindowImage, "Press 'r' to reset street corners",
                          cv.RGB(255,255,0))

        str = "Number of People: %d" % (crowdDensity)
        draw.drawText(densityWindowImage, str, cv.RGB(0,255,0))
        draw.drawText(densityWindowImage, "Input number: " + peoplePerStreetRegStr, cv.RGB(0,255,0), 1)

        modeStr = "Current mode: '" + processMode + "'"
        if trackingRegions['flowWarpMatrix'] is None:
            modeStr = modeStr + " (configuring)"
        if processMode == 'delay':
            secBetweenFrames = delayBetweenFrames / 1000.0
            modeStr = (modeStr + " ( %.03f sec )") % secBetweenFrames
        draw.drawText(stateWindowImage, modeStr, cv.RGB(255,255,0), 0)
        timeString = "Statistic begin: " + time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(firstFrameTime) )
        timeString += " (%.2f min)" % ((currentFrameTime - firstFrameTime) / 60.0)
        draw.drawText(stateWindowImage, timeString, cv.RGB(0,155,0), 1)
        draw.drawText(stateWindowImage, "Time of the frame: " + time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(currentFrameTime) ), cv.RGB(0,155,0), 2)
        draw.drawText(stateWindowImage, "Total number of people: %d" % currPeopleCount, cv.RGB(0,255,0), 3)
        draw.drawText(stateWindowImage, "Frame queue length: %d" % frameQueueLenght, cv.RGB(0,255,0), 4)
        if frame is None:
            draw.drawText(stateWindowImage, "No any frame", cv.RGB(255, 0, 0), 5)

        draw.drawText(stateWindowImage, "'a' - 'auto' mode", cv.RGB(0,155,155), 7)
        draw.drawText(stateWindowImage, "'p' - 'pause' mode ('right' - next, 'up' - 10 next)", cv.RGB(0,155,155), 8)
        draw.drawText(stateWindowImage, "'d' - 'delay' mode (arrows change delay time)", cv.RGB(0,155,155), 9)
        draw.drawText(stateWindowImage, "'q' - Quit", cv.RGB(0,155,155), 10)

        ##################################
        # Update windows
        cv.ShowImage(configWindow, configWindowImage)
        cv.ShowImage(densityWindow, densityWindowImage )
        cv.ShowImage(stateWindow, stateWindowImage )
        cv.ShowImage(statisticWindow, statisticWindowImage)

        # Save current file
        if not processMode in ("pause", "egomotion") or currentFrameTime != previousFrameTime:
            frm = previousFrame
            previousFrame = currentFrame
            currentFrame = frm 
            previousFrameTime = currentFrameTime
            markFrames(currentFrameTime)
            logState(currentFileName, currentFrameTime, crowdDensity, currPeopleCount)

# MAIN
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "app source_image_dir recycle_image_dir"
        sys.exit(1)

    print sys.argv
    sourceDir = sys.argv[1]
    recycleDir = sys.argv[2]
    statFileName = sys.argv[3]
    main(sourceDir, recycleDir, statFileName)
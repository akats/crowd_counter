# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 17:19:38 2012

@author: anatoliy
"""
import os
import os.path as osp
import re

import cv2

# This class should be refactorable to read from a video, one frame at a time
class ImageSequence:
    def __init__(self, sourceDir_, os_ = os):
        # os_ may be a mock filesystem for testing
        self.sourceDir = sourceDir_
        self.os = os_
        self.filePattern = re.compile(r"(\w+).(bmp|BMP|jpg|JPG)")

    def getFileTime(self, filename):
        return float(filename[4:8])# + float(filename[14:17])/1000
    
    def getNextFrame(self, previousFrameTime):
        fileTime = 0
        image = None
        fileList = self.os.listdir(self.sourceDir)
        fileList.sort()
    
        # pattern now in global
        fileName = ""
        for fileName in fileList:
            if self.filePattern.match(fileName):
                fileTime = self.getFileTime(fileName)
                if fileTime > previousFrameTime:
                    filePath=osp.join(self.sourceDir, fileName)
                    #print "File Time: %d", (fileTime)
                    try:
                        image = cv2.imread(filePath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    except IOError: 
                        print "Can't read ", filePath
                        continue
                    break
        return image, fileTime, len(fileList), fileName
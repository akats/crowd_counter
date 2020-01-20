# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 17:19:38 2012

@author: anatoliy
"""
import os
import os.path as osp
import re

import cv2

# This class should be refactorable to read from a video, one frame at a time.
# TODO:  Also, make it an iterator.
class ImageSequence:
    def __init__(self, sourceDir_, recycleDir_, os_ = os):
        # os_ may be a mock filesystem for testing
        self.sourceDir = sourceDir_
        self.recycleDir = recycleDir_
        self.os = os_
        self.filePattern = re.compile(r"(\w+)\.(bmp|BMP|jpg|JPG)")

    def getFileTime(self, filename):
        filename = osp.splitext(filename)[0]
        time = re.search(r"([a-zA-Z_]*)(\d+\.?\d*)", filename).group(2)
        return float(time)
        
    def getNextFrame(self, previousFrameTime):
        fileTime = 0
        image = None
        fileList = self.os.listdir(self.sourceDir)
        fileList.sort()
        print "Previous frame time: {:1.3f}".format(previousFrameTime)
    
        # pattern now in global
        fileName = ""
        for fileName in fileList:
            if self.filePattern.match(fileName):
                fileTime = self.getFileTime(fileName)
                if fileTime > previousFrameTime:
                    filePath=osp.join(self.sourceDir, fileName)
                    #print "File Time: %d", (fileTime)
                    image = cv2.imread(filePath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    if image == None:
                        raise IOError("Can't read filename {0}".format(fileName))
                    return image, fileTime, len(fileList), fileName
        
    # This function will probably go as we redesign our input to take
    # video files, video streams, and file sequences
    def moveProcessedFrames(self, previousFrameTime):
        fileList = os.listdir(self.sourceDir)
        fileList.sort()
        for fileName in self.findProcessedFrames(previousFrameTime, fileList):
            os.rename(osp.join(self.sourceDir, fileName),
                      osp.join(self.recycleDir + fileName))
    
    def findProcessedFrames(self, processedFrameTime, fileList):
        processedFileList = []
        for fileName in fileList:
            if self.filePattern.match(fileName):
                fileTime = self.getFileTime(fileName)
                if fileTime <= processedFrameTime:
                    processedFileList.append(fileName)
        
        return processedFileList
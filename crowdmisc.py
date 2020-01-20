# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:33:41 2012

@author: anatoliy
"""
import cv
import cv2
import numpy as np
#import np.linalg
import time

import cvutils

###############################################################################
## All the parameters and other variables that affect the working of the system
## should go here
###############################################################################
class Properties:
    mode = 'top'
    # TODO:  Make tests to make sure it works in both modes
    egomotion_correction = True
    egomotion_num_corners = 100 # Number of corners to track for egomotion correction
    flow_num_corners = 300 # Number of corners to track for flow computation
    
    # Min and max normalized velocities to count in our average.  These values
    # may be dataset dependent.
    if mode=='velo':
        min_velocity = 0     
        max_velocity = 100
    else:
        min_velocity_in_pixels = 2
        max_velocity_rectified = 0.5
    
    # Is the crowd moving in the horizontal(0) or vertical(1) direction?
    dimension_of_motion = 1
    sample_rect_dims = (0.1, 0.9)
    neighborhood_size_coeff = 1

def timeit(method):

    def timed(*args, **kw):
        verbose = False
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        
        if verbose:
            print '%r %2.2f sec' % \
                  (method.__name__,  te-ts)
        return result

    return timed
    
class TrackedFrame:
    def __init__(self, image, goodFeaturesToTrack = cv2.goodFeaturesToTrack):
        #self.image = cv2.resize(image, (0, 0), fx = 0.25, fy = 0.25)
        self.image = image
        self.features = None
        self.pyramid = [image]
        self.goodFeaturesToTrack = goodFeaturesToTrack
        
    def __eq__(self, other):
        return np.all(self.image == other.image)
    
    def getTrackingFeatures(self, num_features, mask):
        if self.features == None:
            min_distance = np.linalg.norm(self.image.shape)
            min_distance = int(min_distance/100)
            self.features =  self.goodFeaturesToTrack(
                self.image,
                10000, # about the size of the min_distance-length grid
                0.05, min_distance)
        resized_features = np.reshape(np.round(self.features), (-1,2)).astype(np.int32)
        in_mask_idx = np.nonzero(mask[resized_features[:, 1],
                                      resized_features[:, 0]])[0]
        in_mask_idx = in_mask_idx[:num_features]
        return self.features.take(in_mask_idx, 0)
    
    def getImage(self):
        return self.image
    def getShape(self):
        return self.image.shape

# TODO:  Stop messing around and make trackingRegions a class already.
def constructTrackingRegions(baseImage):
    trackingRegions ={\
        'flowCorners': [],
        'displayFlowCorners': [],
        'configImageZoom':0,
        'flowMask': None,
        'flowWarpMatrix': None,
        'stableCorners': [],
        'displayStableCorners':[],
        'stableRegionMask':None,
        'baseImage':baseImage,
        'minVelocity':None,
        'maxVelocity':None
    }
    if type(baseImage) is np.ndarray:
        baseImage = trackingRegions['baseImage'] = TrackedFrame(baseImage)
    if baseImage != None and baseImage.getShape()[0] != 0:
        trackingRegions['flowMask'] =\
            np.zeros(baseImage.getShape(), dtype = np.uint8)
        trackingRegions['stableRegionMask']=\
            np.zeros(baseImage.getShape(), dtype = np.uint8)
    return trackingRegions

def updateEgomotionTrackingRegion(newPoints, configImageZoom, trackingRegions):
    for newPoint in newPoints:
        trackingRegions['displayStableCorners'].append(newPoint)
        trackingRegions['configImageZoom'] = configImageZoom
        newPoint = tuple([int(round(x/configImageZoom)) for x in newPoint])
        trackingRegions['stableCorners'].append(newPoint)
        if len(trackingRegions['stableCorners']) % 4 == 0:
            stableCorners = np.array([trackingRegions['stableCorners'][-4:]],
                                     dtype = np.int32)
            cv2.fillPoly(trackingRegions['stableRegionMask'],\
                        stableCorners,
                        (255, 0, 0))
    
    #print trackingRegions['displayStableCorners']
    return trackingRegions

def updateFlowTrackingRegion(newPoints, configImageZoom, trackingRegions):
    for newPoint in newPoints:
        if len(trackingRegions['flowCorners']) == 4:
            trackingRegions['flowCorners'] = []
            trackingRegions['displayFlowCorners'] = []
        
        trackingRegions['displayFlowCorners'].append(newPoint)
        trackingRegions['configImageZoom'] = configImageZoom
        newPoint = tuple([int(round(x/configImageZoom)) for x in newPoint])
        trackingRegions['flowCorners'].append(newPoint)
        if len(trackingRegions['flowCorners']) == 4:
            trackingRegions['flowMask'][:] = 0
            flowCorners = np.array([trackingRegions['flowCorners']],
                                   dtype = np.int32)
            cv2.fillPoly(trackingRegions['flowMask'],
                         flowCorners, (255, 0, 0, 0))

#TODO:  Make these into a separate function.  When converting to a class,
#update of thesholds should be done whenever flowWarpMatrix is computed.
#These lines are also in warpTrackingRegions.
            dimension_of_motion = Properties.dimension_of_motion
            flowWarpMatrix = trackingRegions['flowWarpMatrix'] = \
                computeRectificationParams(trackingRegions['flowCorners'])
            trackingRegions['minVelocity'] = Properties.min_velocity_in_pixels * \
                flowWarpMatrix[dimension_of_motion, dimension_of_motion]
            trackingRegions['maxVelocity'] = Properties.max_velocity_rectified

    #print trackingRegions['displayFlowCorners']
    return trackingRegions
    
# Compute the warping matrix from the image region of interests(i.e., where 
# people are, to the coordinate system where density is computed)
# This is also future trackingRegion class method.
def computeRectificationParams(boundary_pts):
    mode = Properties.mode
    if mode == 'velo':
        warp_matrix = np.eye(3, dtype=np.float32);
        boundary_pts = np.array(boundary_pts)
        return cv.fromarray(warp_matrix)
    else:
        return computeUnitSquareTransform(boundary_pts)
        
def computeUnitSquareTransform(pts):
    pts=np.array(pts, dtype=np.float32)
    rectified_width=rectified_height=1
    rectified_pts = np.array([0, rectified_height,
                              0, 0,
                              rectified_width, 0,
                              rectified_width, rectified_height])
    rectified_pts = np.reshape(rectified_pts, (-1, 2)).astype(np.float32)
    
    warp_matrix = cv2.findHomography(pts, rectified_pts, 0)[0]
    return warp_matrix

def warpTrackingRegions(trackingRegions, base_image, warp_matrix):
    newTrackingRegions = constructTrackingRegions(base_image)
    
    if len(trackingRegions['stableCorners']) > 0:    
        corners = np.array(trackingRegions['stableCorners'], dtype=np.float32)
        corners = corners.reshape((1, -1, 2))
        warped_corners = np.zeros_like(corners)
        cv2.perspectiveTransform(corners, warp_matrix, warped_corners)
        updateEgomotionTrackingRegion(
            cvutils.array2point_list(warped_corners), 1, newTrackingRegions)

    corners = np.array(trackingRegions['flowCorners'], dtype = np.float32)
    corners = corners.reshape((1, -1, 2))
    warped_corners = cv2.perspectiveTransform(corners, warp_matrix)[0]
    updateFlowTrackingRegion(
        cvutils.array2point_list(warped_corners), 1, newTrackingRegions)
        
    newTrackingRegions['configImageZoom'] = trackingRegions['configImageZoom']
    newTrackingRegions['displayFlowCorners'] =\
        [(int(round(x[0] * newTrackingRegions['configImageZoom'])),
          int(round(x[1] * newTrackingRegions['configImageZoom'])) )
        for x in newTrackingRegions['flowCorners'] ]
    newTrackingRegions['displayStableCorners'] =\
        [(int(round(x[0] * newTrackingRegions['configImageZoom'])),
          int(round(x[1] * newTrackingRegions['configImageZoom'])) )
        for x in newTrackingRegions['stableCorners'] ]
    
    # flowWarpMatrix M satisfies M*flowCorners = unit square.
    dimension_of_motion = Properties.dimension_of_motion
    flowWarpMatrix = newTrackingRegions['flowWarpMatrix'] = \
        computeRectificationParams(newTrackingRegions['flowCorners'])
    newTrackingRegions['minVelocity'] = Properties.min_velocity_in_pixels * \
        flowWarpMatrix[dimension_of_motion, dimension_of_motion]
    newTrackingRegions['maxVelocity'] = Properties.max_velocity_rectified
    
    return newTrackingRegions
    
    

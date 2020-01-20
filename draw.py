# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:15:49 2012

@author: anatoliy
"""
import time

import numpy as np
import cv
import cv2

import cvutils
from cvutils import array2point_list
from crowdmisc import Properties

SmallTextFont = None
LargeTextFont = None
def drawText(image, text, color, rowNumber = 0):
    global SmallTextFont
    global LargeTextFont
    
    if(image.shape[1] > 900):
        TextFont = LargeTextFont
        if TextFont is None:
            TextFont = cv2.FONT_HERSHEY_COMPLEX
            rowHeight = 55
    else:
        TextFont = SmallTextFont
        if TextFont is None:
            TextFont = cv2.FONT_HERSHEY_COMPLEX_SMALL
            rowHeight = 25
        # Ubuntu-Mono-R /usr/share/fonts/truetype
        #TextFont = ImageFont.truetype('/tmp/font.ttf', 25)
    cv2.putText(image, text, (0, rowHeight * (rowNumber + 1)),
               TextFont, 1, color)
    
def drawPoints(image, points, color = cv.RGB(255, 0, 0)):
    if(image.squeeze().ndim == 2):
        image = np.tile(image[:, :, np.newaxis], (1,1,3))
    points = cvutils.array2point_list(points)
    for p in points:
        cv2.circle(image, (int(p[0]), int(p[1])), 3, color, 1)
    return image

def drawLines(image, points1, points2, color = (0, 0, 0)):
    if(image.squeeze().ndim == 2):
        image = np.tile(image[:, :, np.newaxis], (1,1,3))
    points1 = cvutils.array2point_list(points1)
    points2 = cvutils.array2point_list(points2)
    # TODO:  handle exception if different size
    for p1, p2 in zip(points1, points2):
        cv2.line(image, p1, p2, color, thickness=1, lineType=4)
    
    return image

def draw_vertical_sample_boundaries(
    curr_image_cv, warp_matrix, square_length):
    
    #TODO:  IF THIS IS EVER USED, DO NOT USE THIS VALUE
    assert(False)
    scale_factor = 0.5
    
    curr_image = np.array(curr_image_cv)
    full_width = curr_image.shape[1]
    full_height = curr_image.shape[0]
    width = int(curr_image.shape[1] * scale_factor)
    height = int(curr_image.shape[0] * scale_factor)
    
    full_left_bound = int(full_width*(0.5 - square_length/2))
    full_right_bound = int(full_width*(0.5 + square_length/2))
    left_bound = int(full_left_bound * scale_factor)
    right_bound = int(full_right_bound * scale_factor)
    

    full_line1 = ((full_left_bound, 0), (full_left_bound, full_height-1))
    full_line2 = ((full_right_bound, 0), (full_right_bound, full_height-1))    
    line1 = ((left_bound, 0), (left_bound, height-1))
    line2 = ((right_bound, 0), (right_bound, height-1))
    
    if curr_image.ndim == 2:
        sample_image = curr_image[:, :, np.newaxis]
        sample_image = np.tile(sample_image, (1,1,3))
    sample_image = cv.fromarray(sample_image)
    resized_sample_image = cv.CreateMat(height, width, sample_image.type)
    cv.Resize(sample_image, resized_sample_image)
    cv.Line(resized_sample_image, line1[0], line1[1], cv.CV_RGB(0, 255, 0))
    cv.Line(resized_sample_image, line2[0], line2[1], cv.CV_RGB(0, 255, 0))
    boundary_pts = line1 + line2
    
    return resized_sample_image, \
        cvutils.reorder_boundary_points(
            np.array(full_line1 + full_line2))

def backproject_sample_rect(warp_matrix, sample_dims):
    warp_matrix_inv = cv2.invert(warp_matrix)[1]
    #warp_matrix_inv = np.array(warp_matrix_inv)/cv.Get2D(warp_matrix_inv, 2, 2)[0]
    #print warp_matrix_inv
    warp_matrix_inv = cv.fromarray(warp_matrix_inv)
    rectified_shape = np.array((1.,1.));
    width_center = rectified_shape[1]/2;
    sample_dims = [x/2 for x in sample_dims]
    
    #roi_bottom = (rectified_shape[0] - neighborhood_length) * np.random.random() + \
    #    neighborhood_length
    roi_bottom = 0.5
                           
    rectified_sample_roi = [
          width_center - sample_dims[0], roi_bottom + sample_dims[1],
          width_center - sample_dims[0], roi_bottom - sample_dims[1],
          width_center + sample_dims[0], roi_bottom - sample_dims[1],
          width_center + sample_dims[0], roi_bottom + sample_dims[1]]
    rectified_sample_roi = np.array(rectified_sample_roi, dtype=np.float32)
    rectified_sample_roi = rectified_sample_roi.reshape((1,4,2))
    backprojected_sample_boundary = cv.CreateMat(1,4, cv.CV_32FC2)
    rectified_sample_roi = cv.fromarray(rectified_sample_roi)
    cv.PerspectiveTransform(rectified_sample_roi, backprojected_sample_boundary,
                            warp_matrix_inv)
    
    return cvutils.array2point_list(np.array(backprojected_sample_boundary))

def compute_sample_image_coordinates(image_shape, sample_bounds):
    # TODO:  get this coefficient out of here
    # Also, if the sample is not a square, we may amplify the differences by
    # the neighborhood, unnaturally.  Include a maximum number of pixels as
    # a function of screen resolution
    neighborhood_size_coeff=Properties.neighborhood_size_coeff
    
    sample_roi = cv.BoundingRect(sample_bounds, 0)
    neighborhood_roi = (sample_roi[0] - neighborhood_size_coeff * sample_roi[2],
                        sample_roi[1] - neighborhood_size_coeff * sample_roi[3],
                        sample_roi[2] * (2 * neighborhood_size_coeff + 1),
                        sample_roi[3] * (2 * neighborhood_size_coeff + 1) )
    
    #print cv.Get1D(backprojected_neighborhood_boundary, 0)
    neighborhood_roi = cvutils.rect_intersection(neighborhood_roi,
                                                 (0, 0, image_shape[1], image_shape[0]))
    
    replicated_neighborhood_corner = np.hstack((
        neighborhood_roi[0]*np.ones((4,1)), 
        neighborhood_roi[1]*np.ones((4,1))))
    replicated_neighborhood_corner = replicated_neighborhood_corner.reshape((1,4,2))
    sample_bounds = np.array(sample_bounds)
    local_sample_boundary = sample_bounds - replicated_neighborhood_corner
                                             
    return neighborhood_roi, cvutils.array2point_list(local_sample_boundary)

def draw_backprojected_sample_rect( \
    curr_image_cv, warp_matrix, square_length):

    curr_image = np.array(curr_image_cv)
    shape = curr_image.shape
    sample_rect = backproject_sample_rect(warp_matrix, square_length)
    neighborhood_roi, local_sample_shape =\
        compute_sample_image_coordinates(shape, sample_rect)

    sample_image = \
        curr_image[neighborhood_roi[1]:neighborhood_roi[1]+neighborhood_roi[3],
                   neighborhood_roi[0]:neighborhood_roi[0]+neighborhood_roi[2]]
    sample_image = sample_image.squeeze()
    if sample_image.ndim == 2:
        sample_image = sample_image[:, :, np.newaxis]
        sample_image = np.tile(sample_image, (1,1,3))
    
    cv2.polylines(sample_image, np.array([local_sample_shape]), True, cv.RGB(255, 0, 0))

    return sample_image, sample_rect

def draw_sample_region(
        curr_image_cv, warp_matrix, square_length):
    mode = Properties.mode
    if mode == 'velo':
        return draw_vertical_sample_boundaries(
            curr_image_cv, warp_matrix, square_length)
    else:
        return draw_backprojected_sample_rect(
            curr_image_cv, warp_matrix, square_length)
    
statImage = None
def drawResult(currentFrame,
               prevFeatures, currFeatures,
               meanVelocity, velocityInlierIdx, 
               numInImage, numInSample,
               numTotal,
               frameTime, 
               sampleRegionBounds, 
               baseTrackingRegions, trackingRegions,
               prevEgomotionMatrix, currEgomotionMatrix):
    global statImage
    if statImage is None:
         statImage = np.zeros(currentFrame.shape+tuple([3]),
                              dtype = np.uint8)

    # Warp everything to base
    prevEgomotionMatrix = np.linalg.inv(prevEgomotionMatrix)
    currEgomotionMatrix = np.linalg.inv(currEgomotionMatrix)
    currentFrame = cv2.warpPerspective(currentFrame, currEgomotionMatrix,
                                       currentFrame.shape[1::-1])
    prevFeatures = cv2.perspectiveTransform(prevFeatures, prevEgomotionMatrix)
    currFeatures = cv2.perspectiveTransform(currFeatures, currEgomotionMatrix)
    
    if currentFrame.squeeze().ndim == 2:
        statImage = cv2.cvtColor(currentFrame, cv2.COLOR_GRAY2BGR)
    else:
        statImage = currentFrame.copy()
        
    prevFeaturesNP = np.atleast_2d(prevFeatures).reshape(1, -1, 2)
    currFeaturesNP = np.atleast_2d(currFeatures).reshape(1, -1, 2)
    prevInlierFeatures = prevFeaturesNP.take(velocityInlierIdx, 1)
    currInlierFeatures = currFeaturesNP.take(velocityInlierIdx, 1)
    velocityOutlierIdx = np.setdiff1d(np.arange(prevFeaturesNP.shape[1]),
                                      velocityInlierIdx)
    prevOutlierFeatures = prevFeaturesNP.take(velocityOutlierIdx, 1)
    currOutlierFeatures = currFeaturesNP.take(velocityOutlierIdx, 1)

    drawText(statImage, "Time " + time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(frameTime) ), cv.RGB(255, 255, 0), 1)
    drawText(statImage, "Total Seen: %d" % (numTotal), cv.RGB(255, 255, 0), 2)
    #drawText(statImage, "Total Seen: Calculating...", cv.RGB(255, 255, 0), 2)
    densityString = "In this image: %d.  " %(numInImage)
    if sampleRegionBounds != None:
        densityString += "In the sample: %d" %(numInSample)
    drawText(statImage, densityString, cv.RGB(255, 255, 0), 3)
    drawText(statImage, "Velocity: %1.3f milliRegionHeights per sec." % (1e3 * meanVelocity), cv.RGB(255, 255, 0), 4)

    drawPoints(statImage, array2point_list(prevInlierFeatures), cv.RGB(255, 0, 0))
    drawPoints(statImage, array2point_list(currInlierFeatures), cv.RGB(0, 255, 0))
    drawPoints(statImage, array2point_list(currOutlierFeatures), cv.RGB(255, 255, 0))
    drawPoints(statImage, array2point_list(prevOutlierFeatures), cv.RGB(255, 128, 0))
    drawLines(statImage, prevInlierFeatures, currInlierFeatures, (0, 0, 0))
    
    # bound height
    flowCorners = baseTrackingRegions['flowCorners']
    stableCorners = baseTrackingRegions['stableCorners']
    cv2.line(statImage, flowCorners[3], flowCorners[0], cv.CV_RGB(255, 0, 0))
    if sampleRegionBounds != None:
        cv2.polylines(statImage, np.array([sampleRegionBounds]),
                    True, cv.RGB(0, 255, 0))
    # draw the stable regions
    stablePolyLines = zip(*[stableCorners[i::4] for i in range(4)])
    stablePolyLines = np.array([list(x) for x in stablePolyLines], 
                                dtype = np.int32)
    cv2.polylines(statImage, stablePolyLines,
                True, cv.RGB(0,  0, 255))
    return statImage
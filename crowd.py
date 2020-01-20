# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:23:30 2012

Moving crowd counting library

@author: akats
"""

import cv
import cv2
import numpy as np
import numpy.linalg as linalg

import cvutils
from crowdmisc import Properties
mode = Properties.mode
egomotion_num_corners=Properties.egomotion_num_corners
flow_num_corners = Properties.flow_num_corners
dimension_of_motion = Properties.dimension_of_motion
egomotion_correction = Properties.egomotion_correction
import crowdmisc

velocities_log = []

# approximate the location of the region of interest in the next image
def update_boundary_pts(curr_image, egomotion_matrix,
                        prev_trackpts, curr_trackpts,
                        trackingRegions):
    mode = Properties.mode
    if mode == 'velo':
        #TODO:  Test this
        trackingRegions['flowCorners'] = correct_horizontal_motion(
            curr_image, prev_trackpts, curr_trackpts, trackingRegions['flowCorners'])
    else:
        pass
    
    trackingRegions['displayFlowCorners'] = \
        [(int(x[0] * trackingRegions['configImageZoom']),
         int(x[1] * trackingRegions['configImageZoom']))
         for x in trackingRegions['flowCorners']]
     
    return trackingRegions
         

# For velo mode, the boundaries are two lines in the direction of motion that
# delineate the location of the people.  One goes along the presumed heads,
# the other along the presumed feet.  This corrects the location based
# on camera egomotion.  This code has never been tested in production.
def correct_horizontal_motion(image, prev_trackpts, curr_trackpts, boundary_pts):
    image_boundaries = (1, 1, image.width-2, image.height-2)
    motion_matrix=cv.CreateMat(3, 3, cv.CV_32FC1)
    cv.FindHomography(prev_trackpts, curr_trackpts,
                      motion_matrix,
                      cv.CV_RANSAC, 10)
    boundary_pts = boundary_pts.reshape((1, -1, 2)).astype(np.float32)
    warped_boundary_pts = np.zeros_like(boundary_pts)
    cv.PerspectiveTransform(boundary_pts, warped_boundary_pts, motion_matrix)
    pt11, pt12 = cvutils.line_rectangle_intersection(
        tuple(warped_boundary_pts[0, 0,:]), tuple(warped_boundary_pts[0, 3,:]),
        image_boundaries)
    pt21, pt22 = cvutils.line_rectangle_intersection(
        tuple(warped_boundary_pts[0, 1,:]), tuple(warped_boundary_pts[0, 2,:]),
        image_boundaries)
    
    boundary_pts = cvutils.reorder_boundary_points(np.array((pt11, pt12, pt21, pt22)))
    return boundary_pts.reshape((-1, 2)).astype(np.int), motion_matrix

# TODO:  rename.  This isn't really correcting anything, and find_egomotion
# is already taken
def correct_egomotion(prevTrackedFrame, currTrackedFrame, baseTrackingRegions):
    curr_egomotion_matrix = None
    # mode = Properties.mode # is global necessary here?  We are importing the variable.
    if mode == 'top':
        base_egomotion_matrix = find_egomotion(baseTrackingRegions['baseImage'],
                                      prevTrackedFrame, baseTrackingRegions,
                                      None, None)
        base_egomotion_matrix = base_egomotion_matrix[0]
        #print np.array(base_egomotion_matrix)

        prevTrackingRegions = crowdmisc.warpTrackingRegions(
            baseTrackingRegions, prevTrackedFrame, base_egomotion_matrix)
        
        curr_egomotion_matrix = find_egomotion(prevTrackedFrame, currTrackedFrame,
                                               prevTrackingRegions,
                                               None, None)
        curr_egomotion_matrix = curr_egomotion_matrix[0]
    else:
        curr_egomotion_matrix = np.eye(3, dtype=np.float32)
        prevTrackingRegions = baseTrackingRegions
        
    return base_egomotion_matrix, curr_egomotion_matrix, prevTrackingRegions

def find_egomotion(prevTrackedFrame, currTrackedFrame, trackingRegion,
                   prev_trackpts, curr_trackpts):
    prev_anchorpts = None
    curr_anchorpts = None
    if egomotion_correction == False:
        egomotion_matrix=np.eye(3, dtype=np.float32)
        return egomotion_matrix, None, None
    elif mode == 'velo':
        prev_anchorpts = prev_trackpts
        curr_anchorpts = curr_trackpts
    else:
        # TODO:  Put a warning if the spatial distribution of the input anchor
        # points differs significantly from that of the output anchor points
        prev_anchorpts = prevTrackedFrame.getTrackingFeatures(
            egomotion_num_corners, trackingRegion['stableRegionMask'])
        curr_anchorpts, is_found, track_error= cv2.calcOpticalFlowPyrLK( 
            prevTrackedFrame.getImage(), currTrackedFrame.getImage(),
            prev_anchorpts,
            winSize = (16, 16), maxLevel = 6, 
            criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 0.003),
            flags = 0)
        curr_anchorpts_np = np.array(curr_anchorpts, np.float32)
        prev_anchorpts_np = np.array(prev_anchorpts, np.float32)
        is_found = np.array(is_found)
        curr_anchorpts_np = curr_anchorpts_np[np.nonzero(is_found)]
        prev_anchorpts_np = prev_anchorpts_np[np.nonzero(is_found)]
        prev_anchorpts = prev_anchorpts_np.reshape(1, -1, 2)
        curr_anchorpts = curr_anchorpts_np.reshape(1, -1, 2)

    egomotion_matrix, ransac_inliers = cv2.findHomography(
        prev_anchorpts, curr_anchorpts, cv2.RANSAC, 1)

    ransac_inliers = np.nonzero(np.array(ransac_inliers).ravel())[0]
    # TODO:  Count the number of points that are actually not far from their
    # matches, instead of relying on ransac_inliers.  Use a different threshold
    # since the one in FindHomography may be too restrictive, and distort
    # the number of actual matches.
    if(ransac_inliers.shape[0] * 2 < egomotion_num_corners):
        print "Warning:  Less then half the trackpoints are used in egomotion computation"
    curr_anchorpts = curr_anchorpts[:, ransac_inliers, :]
    prev_anchorpts = prev_anchorpts[:, ransac_inliers, :]
    return egomotion_matrix, \
           prev_anchorpts_np.reshape(-1, 2), \
           curr_anchorpts_np.reshape(-1, 2)
    
# For a stationary camera, find which features attach themselves to stationary
# objects instead of people.  This function is not tested.
def update_stationary_points(prev_histogram, new_features, learning_rate):
    new_distribution = np.zeros_like(prev_histogram)
    for point in new_features:
        cv.Circle(new_distribution, point, 2, cv.CV_RGB(255, 255, 255), -1)
    
    cv.RunningAvg(new_distribution, prev_histogram, learning_rate)
    return prev_histogram

def calculate_density(people_in_sample, sample_rect_dims):
    mode = Properties.mode
    if mode == 'velo':
        return float(people_in_sample)/sample_rect_dims[0]
    else:
        return float(people_in_sample)/(sample_rect_dims[0] * sample_rect_dims[1])

@crowdmisc.timeit
def compute_pedestrian_flow(prevTrackedFrame, currTrackedFrame,
                            trackingRegions,
                            density, prev_count):
    base_egomotion_matrix, egomotion_matrix, prevTrackingRegions = \
        correct_egomotion(prevTrackedFrame, currTrackedFrame, trackingRegions)

    velocity, prev_features, curr_features , velocity_inlier_idx= \
        compute_horizontal_speed(prevTrackedFrame, currTrackedFrame,
                                 egomotion_matrix, prevTrackingRegions)
    currTrackingRegions = crowdmisc.warpTrackingRegions(
        prevTrackingRegions, currTrackedFrame, egomotion_matrix)

    trackingRegions = update_boundary_pts(currTrackedFrame.getImage(),
                                          egomotion_matrix,
                                          prev_features, curr_features,
                                          trackingRegions)

    if mode=='velo':
        velocity /= prevTrackedFrame.getShape()

    curr_count = prev_count + density * velocity
    
    print "Density: %1.1f; Velocity: %1.3f" % (density, velocity)
    return curr_count, prev_features, curr_features, \
           velocity, velocity_inlier_idx, \
           currTrackingRegions, \
           base_egomotion_matrix, np.dot(base_egomotion_matrix, egomotion_matrix)
    
def compute_horizontal_speed(prevTrackedFrame, currTrackedFrame,\
                             egomotion_matrix,\
                             trackingRegions):
    boundary_pts = trackingRegions['flowCorners']
    warp_matrix = np.matrix(trackingRegions['flowWarpMatrix'])
    egomotion_matrix = np.matrix(egomotion_matrix)
    assert(prevTrackedFrame.getImage().squeeze().ndim == 2)
    assert(currTrackedFrame.getImage().squeeze().ndim == 2)

    # Parameters
    num_corners_to_track = flow_num_corners;
    
    mask = np.zeros(prevTrackedFrame.getShape(), dtype=np.uint8)
    cv2.fillPoly(mask, np.array([boundary_pts], dtype = np.int32), (255, 0, 0));
    
    prev_features = prevTrackedFrame.getTrackingFeatures(
        num_corners_to_track, mask)
    curr_features = cv2.perspectiveTransform(prev_features, egomotion_matrix)
    curr_features, is_found, track_error= cv2.calcOpticalFlowPyrLK( 
        prevTrackedFrame.getImage(), currTrackedFrame.getImage(),
        prev_features,
        nextPts = curr_features,
        winSize = (32, 32), maxLevel = 7, 
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 200, 0.003),
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW)

    is_found = [x[0] if x[1]<1500 else 0 for x in zip(is_found, track_error)]
    curr_features = curr_features.take(np.nonzero(is_found)[0], 0)
    prev_features = prev_features.take(np.nonzero(is_found)[0], 0)
    
    rectified_prev_features = cv2.perspectiveTransform(
        prev_features, warp_matrix)
    warp_matrix = warp_matrix * linalg.inv(egomotion_matrix)
    rectified_curr_features = cv2.perspectiveTransform(
        curr_features, warp_matrix)
        
    velocities = curr_features[:, :, dimension_of_motion] -\
                 prev_features[:, :, dimension_of_motion]
    velocities = np.sum(velocities**2, 0)

    rectified_velocities=rectified_curr_features[:,:,dimension_of_motion] - \
                         rectified_prev_features[:,:,dimension_of_motion]
    inlier_idx = np.nonzero(np.logical_and(
        velocities > Properties.min_velocity_in_pixels**2,
        rectified_velocities < Properties.max_velocity_rectified))[0] 
    if(inlier_idx.shape[0]<0.05 * num_corners_to_track):
        mean_velocity = 0
    else:
        mean_velocity = np.mean(rectified_velocities[inlier_idx])
    #print "mean velocity: %1.3e" % mean_velocity
    velocities_log.append((prev_features, curr_features))

    return mean_velocity, prev_features, curr_features, inlier_idx
    
#def increment_people_counter(prev_counter, velocity, density):
#    return prev_counter + density*rectified_height*velocity
    
#return np.zeros((100, 100))

def showResizedImage(image, size, name):
    resized_image = cv.CreateMat(size[0], size[1], image.type)
    cv.NamedWindow(name)
    cv.Resize(image, resized_image)
    cv.ShowImage(name, resized_image)


if __name__ == '__main__':
    #test_sample_image()
    test_find_egomotion()
#test_compute_rectification_params()
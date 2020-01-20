# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 17:23:30 2012

Moving crowd counting library

@author: akats
"""

import cv
import numpy as np
import numpy.linalg as linalg
import matplotlib as mpl
from matplotlib import pyplot as plt
from PIL import Image

import cvutils
from crowdmisc import mode, egomotion_num_corners, flow_num_corners,\
min_velocity, max_velocity, dimension_of_motion
import crowdmisc
import draw #this is temporary.

# approximate the location of the region of interest in the next image
def update_boundary_pts(curr_image, egomotion_matrix,
                        prev_trackpts, curr_trackpts,
                        trackingRegions):
    global mode
    if mode == 'velo':
        #TODO:  Test this
        trackingRegions['flowCorners'] = correct_horizontal_motion(
            curr_image, prev_trackpts, curr_trackpts, boundary_pts)
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
def correct_egomotion(prev_image, curr_image, baseTrackingRegions):
    curr_egomotion_matrix = None
    # global mode # is global necessary here?  We are importing the variable.
    if mode == 'top':
        base_egomotion_matrix = find_egomotion(baseTrackingRegions['baseImage'],
                                      prev_image, baseTrackingRegions,
                                      None, None)
        base_egomotion_matrix = base_egomotion_matrix[0]
        #print np.array(base_egomotion_matrix)

        prevTrackingRegions = crowdmisc.warpTrackingRegions(
            baseTrackingRegions, prev_image, base_egomotion_matrix)
        
        curr_egomotion_matrix = find_egomotion(prev_image, curr_image,
                                               prevTrackingRegions,
                                               None, None)
        curr_egomotion_matrix = curr_egomotion_matrix[0]
    else:
        curr_egomotion_matrix = cv.Identity(cv.CreateMat(3,3, CV_32FC1))
        prevTrackingRegions = baseTrackingRegions
        
    return curr_egomotion_matrix, prevTrackingRegions

def find_egomotion(prev_image, curr_image, trackingRegion, prev_trackpts, curr_trackpts):
    prev_anchorpts = None
    curr_anchorpts = None
    num_corners_to_track = egomotion_num_corners
    if crowdmisc.egomotion_correction == False:
        egomotion_matrix=cv.CreateMat(3, 3, cv.CV_32FC1)
        cv.SetIdentity(egomotion_matrix)
        return egomotion_matrix, None, None
    elif mode == 'velo':
        prev_anchorpts = prev_trackpts
        curr_anchorpts = curr_trackpts
    else:
        # TODO:  Put a warning if the spatial distribution of the input anchor
        # points differs significantly from that of the output anchor points
        prev_anchorpts = cv.GoodFeaturesToTrack(
            prev_image, None, None, num_corners_to_track, 0.05, 30,
            trackingRegion['stableRegionMask'])
        curr_anchorpts, is_found, track_error= cv.CalcOpticalFlowPyrLK( 
            prev_image, curr_image,
            None, None,
            prev_anchorpts,
            (16, 16), 6, 
            (cv.CV_TERMCRIT_EPS+cv.CV_TERMCRIT_ITER, 200, 0.003),
            0)
        curr_anchorpts_np = np.array(curr_anchorpts, np.float32)
        prev_anchorpts_np = np.array(prev_anchorpts, np.float32)
        is_found = np.array(is_found)
        curr_anchorpts_np = curr_anchorpts_np[np.nonzero(is_found)]
        prev_anchorpts_np = prev_anchorpts_np[np.nonzero(is_found)]
        prev_anchorpts_np = prev_anchorpts_np.reshape(1, -1, 2)
        curr_anchorpts_np = curr_anchorpts_np.reshape(1, -1, 2)
            
        curr_anchorpts = cv.fromarray(curr_anchorpts_np)
        prev_anchorpts = cv.fromarray(prev_anchorpts_np)
    
    ransac_inliers = cv.CreateMat(1, prev_anchorpts.cols, cv.CV_8UC1)
    egomotion_matrix=cv.CreateMat(3, 3, cv.CV_32FC1)
    cv.Zero(egomotion_matrix)
    cv.FindHomography(prev_anchorpts, curr_anchorpts,
                      egomotion_matrix,
                      cv.CV_RANSAC, 1, ransac_inliers)

    ransac_inliers = np.nonzero(np.array(ransac_inliers).ravel())[0]
    # TODO:  Count the number of points that are actually not far from their
    # matches, instead of relying on ransac_inliers.  Use a different threshold
    # since the one in FindHomography may be too restrictive, and distort
    # the number of actual matches.
    if(ransac_inliers.shape[0] * 2 < num_corners_to_track):
        print "Warning:  Less then half the trackpoints are used in egomotion computation"
    curr_anchorpts_np = curr_anchorpts_np[:, ransac_inliers, :]
    prev_anchorpts_np = prev_anchorpts_np[:, ransac_inliers, :]
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
    global mode
    if mode == 'velo':
        return float(people_in_sample)/sample_rect_dims[0]
    else:
        return float(people_in_sample)/(sample_rect_dims[0] * sample_rect_dims[1])

def compute_pedestrian_flow(prev_image, curr_image,
                            trackingRegions,
                            density, prev_count):
    egomotion_matrix, prevTrackingRegions = \
        correct_egomotion(prev_image, curr_image, trackingRegions)

    velocity, prev_features, curr_features , velocity_inlier_idx= \
        compute_horizontal_speed(prev_image, curr_image,
                                 egomotion_matrix, prevTrackingRegions)
    currTrackingRegions = crowdmisc.warpTrackingRegions(prevTrackingRegions,
                                                        curr_image,
                                                        egomotion_matrix)

    trackingRegions = update_boundary_pts(curr_image, egomotion_matrix,
                                          prev_features, curr_features,
                                          trackingRegions)

    if mode=='velo':
        velocity /= prev_image.width

    curr_count = prev_count + density * velocity
    
    print "Density: %1.1f; Velocity: %1.3f" % (density, velocity)
    return curr_count, prev_features, curr_features, \
           velocity, velocity_inlier_idx, \
           currTrackingRegions
    
    
def compute_horizontal_speed(prev_image_cv, curr_image_cv,\
                             egomotion_matrix,\
                             trackingRegions):
    boundary_pts = trackingRegions['flowCorners']
    warp_matrix = np.matrix(trackingRegions['flowWarpMatrix'])
    egomotion_matrix = np.matrix(egomotion_matrix)
    assert(prev_image_cv.channels == 1)
    assert(curr_image_cv.channels == 1)
    global mode

    # Parameters
    global flow_num_corners_to_track
    num_corners_to_track = flow_num_corners;
    
    mask = cv.CreateMat(prev_image_cv.rows, prev_image_cv.cols, cv.CV_8UC1)
    cv.SetZero(mask)
    cv.FillPoly(mask, [boundary_pts], (255, 0, 0, 0));
    
    prev_features = cv.GoodFeaturesToTrack(
        prev_image_cv, None, None, num_corners_to_track, 0.05, 30, mask)
    
    curr_features, is_found, track_error= cv.CalcOpticalFlowPyrLK( 
        prev_image_cv, curr_image_cv,
        None, None,
        prev_features,
        (16, 16), 7, 
        (cv.CV_TERMCRIT_EPS+cv.CV_TERMCRIT_ITER, 200, 0.003),
        0)

    # Prune points with track error too large
    is_found = [x[0] if x[1]<1500 else 0 for x in zip(is_found, track_error)]
    curr_features_np = np.array(curr_features, np.float32)
    prev_features_np = np.array(prev_features, np.float32)
    curr_features_np = curr_features_np[np.nonzero(is_found)]
    prev_features_np = prev_features_np[np.nonzero(is_found)]
    prev_features_np = prev_features_np.reshape(1, -1, 2)
    curr_features_np = curr_features_np.reshape(1, -1, 2)
    curr_features = cv.fromarray(curr_features_np)
    prev_features = cv.fromarray(prev_features_np)
    
    #warp_matrix = cv.fromarray(warp_matrix)
    rectified_prev_features = cv.CreateMat(
        prev_features.rows, prev_features.cols, prev_features.type)
    rectified_curr_features = cv.CreateMat(
        prev_features.rows, prev_features.cols, prev_features.type)
    cv.PerspectiveTransform(
        prev_features, rectified_prev_features,cv.fromarray(warp_matrix))
    warp_matrix = warp_matrix * linalg.inv(egomotion_matrix)
    cv.PerspectiveTransform(
        curr_features, rectified_curr_features, cv.fromarray(warp_matrix))
        
    rectified_prev_features_np = np.array(rectified_prev_features)
    rectified_curr_features_np = np.array(rectified_curr_features)
    velocities=rectified_curr_features_np[:,:,dimension_of_motion] - \
               rectified_prev_features_np[:,:,dimension_of_motion]
    
    inlier_idx = np.nonzero(np.logical_and(
        velocities > min_velocity, velocities<max_velocity))[1]
    if(inlier_idx.shape[0]<0.05 * num_corners_to_track):
        mean_velocity = 0
    else:
        mean_velocity = np.mean(velocities[:, inlier_idx])
    #print "mean velocity: %1.3e" % mean_velocity
    
    return mean_velocity, prev_features, curr_features, inlier_idx
    
#def increment_people_counter(prev_counter, velocity, density):
#    return prev_counter + density*rectified_height*velocity
    
#return np.zeros((100, 100))
        
def test_sample_image():
    curr_image = cv.LoadImageM('test2.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
    #curr_image = curr_image.convert('L')
    #curr_image = np.array(curr_image)
    #boundary_pts = np.array([402, 1306, 816, 526, 1562, 467, 2112, 1322])
    boundary_pts = np.array(\
    [[ 152.,  788.],
    [ 460. , 120.],
    [ 988.  ,124.],
    [ 752.  ,904.]]
    )
    boundary_pts = boundary_pts.reshape((-1, 2));

    warp_matrix, rectified_pts = compute_rectification_params(
        boundary_pts)       
    
    sample_image = sample_image_for_density(curr_image,
                                            warp_matrix, 0.05)
    plt.imshow(sample_image)
    plt.show()
    return sample_image

def showResizedImage(image, size, name):
    resized_image = cv.CreateMat(size[0], size[1], image.type)
    cv.NamedWindow(name)
    cv.Resize(image, resized_image)
    cv.ShowImage(name, resized_image)



# I believe this code is now useless -AK, Jun 18 2012
def test_compute_rectification_params():
    #curr_image = cv.LoadImageM('images/im_1328312054_140.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
    prev_image = cv.LoadImageM('frame1.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
    curr_image = cv.LoadImageM('frame2.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
    curr_image_color = cv.LoadImageM('frame2.jpg', cv.CV_LOAD_IMAGE_COLOR)

    boundary_pts = np.array(\
    [[ 152.,  788.],
    [ 460. , 120.],
    [ 988.  ,124.],
    [ 752.  ,904.]]
    )
    boundary_pts = boundary_pts.reshape((-1, 2));

    warp_matrix, boundary_pts2 = compute_rectification_params(
        boundary_pts)       
    
    mean_velocity, prev_features, curr_features, inlier_idx = \
        compute_horizontal_speed(
            prev_image, curr_image, warp_matrix, boundary_pts2)
    
    out_length = 300
    rectified_image = cv.CreateMat(out_length, out_length, cv.CV_8UC1)
    zoom_matrix = np.eye(3,3)
    zoom_matrix[0,0]=zoom_matrix[1,1]=out_length
    zoom_matrix = np.dot(zoom_matrix, np.array(warp_matrix))
    zoom_matrix_cv = cv.fromarray(zoom_matrix)
    cv.WarpPerspective(curr_image, rectified_image, zoom_matrix_cv)

    prev_features_np = np.array(prev_features)
    curr_features_np = np.array(curr_features)
    draw.drawPoints(curr_image_color, prev_features_np, cv.CV_RGB(255, 0, 0))
    draw.drawPoints(curr_image_color, curr_features_np, cv.CV_RGB(0, 255, 0))
    showResizedImage(curr_image_color, (600, 800), "curr_image")
    showResizedImage(prev_image, (600, 800), "prev_image")
    
    print inlier_idx
    return mean_velocity, prev_features, curr_features, inlier_idx

def find_horizontal_lines_idx(lines):
    x=lines[:, 2] - lines[:, 0];
    y=lines[:, 3] - lines[:, 1];
    return (np.nonzero(np.abs(x)>np.abs(y)))[0]
    
#def test_update_stationary_points():
#    prev_histogram = np.zeros((20, 20), np.float32
#    prev_histogram[9:12, 9:12]=255;
#    features = [(3, 3), (5, 15)];
#    learning_rate = 0.1
#    updated_histogram = update_stationary_points(prev_histogram,
#                                                 features,
#                                                 learning_rate)
#    print updated_histogram


if __name__ == '__main__':
    #test_sample_image()
    test_find_egomotion()
#test_compute_rectification_params()
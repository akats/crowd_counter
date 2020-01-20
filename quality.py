# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:35:30 2012
This module helps examine the quality of output of computer vision and other
mathematical estimation functions.  Whenever possible it creates a formal test
by "memorizing" the correct value, and the SVN checkin of the code on which it
was last tested.
@author: anatoliy
"""

import unittest

import cv
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import scipy.linalg as linalg

import draw
import crowd
import crowdmisc
import cvutils

class TestVision(unittest.TestCase):
    def setUp(self):
        crowdmisc.mode == 'top'
        crowdmisc.egomotion_num_corners = 100 # Number of corners to track for egomotion correction
        crowdmisc.flow_num_corners = 100 # Number of corners to track for flow computation
        
        # Min and max normalized velocities to count in our average.  These values
        # may be dataset dependent.
        if crowdmisc.mode=='velo':
            crowdmisc.min_velocity = 0     
            crowdmisc.max_velocity = 100
        else:
            crowdmisc.min_velocity = 1e-3
            crowdmisc.max_velocity = 2e-1
        
        # Is the crowd moving in the horizontal(0) or vertical(1) direction?
        crowdmisc.dimension_of_motion = 1
        
        self.prevImage = cv.LoadImageM('Jun12/DSC_5817.JPG',
                                   cv.CV_LOAD_IMAGE_GRAYSCALE)
        self.currImage = cv.LoadImageM('Jun12/DSC_5818.JPG',
                                   cv.CV_LOAD_IMAGE_GRAYSCALE)
        self.trackingRegions = crowdmisc.constructTrackingRegions(self.prevImage);
        newPoints = [(0, 700), (0, 0), (175, 0), (175, 700),
                    (1550, 600), (900, 0), (2136, 0), (2136, 700)]
        crowdmisc.updateEgomotionTrackingRegion(newPoints, 1, self.trackingRegions)
        newPoints = [(230, 882), (217, 441), (500, 429), (698, 889)]
        crowdmisc.updateFlowTrackingRegion(newPoints, 1, self.trackingRegions)
    
    def tearDown(self):
        reload(crowdmisc)

    def testComputeHorizontalSpeed(self):
        pi = np.pi
        shear = .95
        theta = pi*3.0/180
        dx = 20
        dy = -30
        warp_matrix = np.array([[shear * np.cos(theta), np.sin(theta), dx],
                                [-np.sin(theta), np.cos(theta), dy],
                                [0, 0, 1] ], dtype = np.float32)
        warp_matrix = cv.fromarray(warp_matrix)
        warped_image = cv.CreateMat(self.currImage.rows, self.currImage.cols,
                                    self.currImage.type)
        cv.WarpPerspective(self.currImage, warped_image, warp_matrix)
        egomotion_matrix1 = np.eye(3)
        egomotion_matrix2, prev_pts2, curr_pts2 = crowd.find_egomotion(
            self.prevImage, self.currImage, self.trackingRegions, None, None)
        egomotion_matrix3, prev_pts3, curr_pts3 = crowd.find_egomotion(
            self.prevImage, warped_image, self.trackingRegions, None, None)
        egomotion_matrix4, prev_pts4, curr_pts4 = crowd.find_egomotion(
            self.currImage, warped_image, self.trackingRegions, None, None)
        #print np.array(egomotion_matrix3)
        #print np.dot(warp_matrix, np.array(egomotion_matrix2))
        #print np.array(warp_matrix)
        #print np.array(egomotion_matrix4)
        
        speed_no_correction = crowd.compute_horizontal_speed(
            self.prevImage, self.currImage,
            egomotion_matrix1,
            self.trackingRegions)
        speed_natural_correction = crowd.compute_horizontal_speed(
            self.prevImage, self.currImage,
            egomotion_matrix2,
            self.trackingRegions)
        speed_warped_image = crowd.compute_horizontal_speed(
            self.prevImage, warped_image,
            egomotion_matrix3,
            self.trackingRegions)                                                
        
        self.assertAlmostEqual(speed_no_correction[0], 0.00521129068702)
        self.assertLess(
            (speed_natural_correction[0] - speed_warped_image[0])/speed_natural_correction[0],
            0.02)
                        
    def test_compute_pedestrian_flow(self):
        base_image = cv.LoadImageM('Jun12/DSC_5156.JPG',
                               cv.CV_LOAD_IMAGE_GRAYSCALE)
        displayFlowCorners = [(267, 449), (266, 205), (411, 198), (544, 450)]
        displayStableCorners =\
            [(5, 269), (6, 3), (231, 5), (233, 309),\
             (1058, 312), (670, 74), (684, 7), (1065, 8), 
             (62, 628), (9, 302), (164, 325), (153, 489)]
        baseTrackingRegions = crowdmisc.constructTrackingRegions(base_image)
        crowdmisc.updateEgomotionTrackingRegion(displayStableCorners, 1,
                                                baseTrackingRegions)
        crowdmisc.updateFlowTrackingRegion(displayFlowCorners, 1,
                                           baseTrackingRegions)
        crowdmisc.updateFlowTrackingRegion(displayFlowCorners, 0.5, baseTrackingRegions)
       
        curr_count, prev_features, curr_features,\
        velocity, velocity_inlier_idx, currTrackingRegions =\
        crowd.compute_pedestrian_flow(self.prevImage, self.currImage,
                                      baseTrackingRegions,
                                      10, 100)

        self.assertAlmostEqual(curr_count, 100.03796491659048)
        self.assertAlmostEqual(velocity, 0.00379649165905)
        npt.assert_equal(
            velocity_inlier_idx,
            np.array([ 0,  3,  5,  9, 10, 11, 18, 21, 23, 25, 26, 29, 35, 36, 46, 47, 48,
                      51, 53, 55, 62, 63, 64, 66, 75, 83, 87, 88, 89, 90, 91, 95, 99]))
        self.assertListEqual(currTrackingRegions['flowCorners'],
                             [(239, 861), (229, 390), (524, 374), (777, 843)])
        self.assertListEqual(
            currTrackingRegions['stableCorners'],
            [(-346, 248), (-359, -36), (-105, -30), (-91, 289),\
            (758, 291), (368, 51), (383, -19), (770, -10),\
            (-264, 614), (-339, 282), (-166, 306), (-171, 474)])

def main_simulation_key_iterator():
    import main
    displayFlowCorners = [(267, 449), (266, 205), (411, 198), (544, 450)]
    for point in displayFlowCorners:
        main.on_mouse(cv.CV_EVENT_LBUTTONDOWN, point[0], point[1], None, None)
        yield ord(' ')
    
    yield ord('e')
    displayStableCorners =\
    [(5, 269), (6, 3), (231, 5), (233, 309),\
     (1058, 312), (670, 74), (684, 7), (1065, 8), 
     (62, 628), (9, 302), (164, 325), (153, 489)]
    for point in displayStableCorners:
        main.on_mouse(cv.CV_EVENT_LBUTTONDOWN, point[0], point[1], None, None)
        yield ord(' ')

    yield 82
    for i in range(10):
        yield ord(' ')
    yield 82
    for i in range(10):
        yield ord(' ')
    yield 82
    for i in range(10):
        yield ord(' ')
    yield 82
    for i in range(10):
        yield ord(' ')
        
    yield ord('q')
    
def simulate_main():
    import main
    key_iterator = main_simulation_key_iterator()
    def waitKey(delayTime):
        return key_iterator.next()
    try:
        main.main('/home/anatoliy/crowd/Jun12/',
                  '/home/anatoliy/crowd/recycled/',
                  '/home/anatoliy/Jun12_result/im',
                  waitKey)
    except:
        cv.DestroyAllWindows()
        cv.NamedWindow('asdf')
        raise

def test_find_egomotion():
    prev_image = cv.LoadImageM('Jun12/DSC_5817.JPG', cv.CV_LOAD_IMAGE_GRAYSCALE)
    curr_image = cv.LoadImageM('Jun12/DSC_5818.JPG', cv.CV_LOAD_IMAGE_GRAYSCALE)
    trackingRegions = crowdmisc.constructTrackingRegions(
        image_size=(prev_image.rows, prev_image.cols));
    for newPoint in [(0, 700), (0, 0), (175, 0), (175, 700),
                     (1550, 600), (900, 0), (2136, 0), (2136, 700)]:
        crowdmisc.updateEgomotionTrackingRegion(newPoint, 1, trackingRegions)
    
    plt.imshow(trackingRegions['stableRegionMask'])
    plt.title('Stable mask')
    
    egomotion_matrix, prev_pts, curr_pts = \
        crowd.find_egomotion(prev_image, curr_image, trackingRegions, None, None)
    warped_image = cv.CreateMat(prev_image.rows, prev_image.cols, prev_image.type)
    cv.Zero(warped_image)
    cv.WarpPerspective(prev_image, warped_image, egomotion_matrix)
    plt.figure()
    plt.imshow(warped_image, 'gray')
    plt.title('Warped image')
    
    curr_image = np.tile(np.array(curr_image)[:, :, np.newaxis], (1,1,3))
    draw.drawPoints(curr_image, cvutils.array2point_list(curr_pts))
    plt.figure()
    plt.imshow(curr_image)
    plt.title('Real current image')
    
if __name__ == '__main__':
    unittest.main()
    #simulate_main()
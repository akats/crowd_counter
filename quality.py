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
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import scipy.linalg as linalg

import draw
import crowd
import crowdmisc
from crowdmisc import Properties
import cvutils

class TestVision(unittest.TestCase):
    def setUp(self):
        Properties.mode == 'top'
        Properties.egomotion_num_corners = 100 # Number of corners to track for egomotion correction
        Properties.flow_num_corners = 100 # Number of corners to track for flow computation
        
        # Min and max normalized velocities to count in our average.  These values
        # may be dataset dependent.
        if Properties.mode=='velo':
            Properties.min_velocity = 0     
            Properties.max_velocity = 100
        else:
            Properties.min_velocity_in_pixels = 1
            Properties.max_velocity = 0.1
        
        # Is the crowd moving in the horizontal(0) or vertical(1) direction?
        Properties.dimension_of_motion = 1
        Properties.sample_rect_dims = (0.1, 0.1)
        
        # pronounced crowd motion, about 10 pixels, almost no camera motion
        self.im5244 = cv2.imread('test_images/DSC_5244.JPG',
                                   cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.im5245 = cv2.imread('test_images/DSC_5245.JPG',
                                   cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # pronounced camera motion, crowd nearly stationary
        self.im5817 = cv2.imread('test_images/DSC_5817.JPG',
                                   cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.im5818 = cv2.imread('test_images/DSC_5818.JPG',
                                   cv2.CV_LOAD_IMAGE_GRAYSCALE)
                                   
        self.frame5244 = crowdmisc.TrackedFrame(self.im5244)
        self.frame5245 = crowdmisc.TrackedFrame(self.im5245)        
        self.frame5817 = crowdmisc.TrackedFrame(self.im5817)
        self.frame5818 = crowdmisc.TrackedFrame(self.im5818)
        
        self.trackingRegions5817 = crowdmisc.constructTrackingRegions(self.im5817);
        newPoints = [(0, 700), (0, 0), (175, 0), (175, 700),
                    (1550, 600), (900, 0), (2136, 0), (2136, 700)]
        crowdmisc.updateEgomotionTrackingRegion(newPoints, 1,
                                                self.trackingRegions5817)
        newPoints = [(230, 882), (217, 441), (500, 429), (698, 889)]
        crowdmisc.updateFlowTrackingRegion(newPoints, 1,
                                           self.trackingRegions5817)
        
        self.trackingRegions5244 = crowdmisc.constructTrackingRegions(self.im5244);
        displayFlowCorners = [(277, 439), (276, 195), (421, 188), (554, 440)]
        displayStableCorners =\
            [(15, 259), (16, 3), (241, 5), (243, 299),\
             (1068, 302), (680, 64), (694, 7), (1075, 8), 
             (72, 618), (19, 292), (174, 315), (163, 479)]
        crowdmisc.updateEgomotionTrackingRegion(displayStableCorners, 0.5,
                                                self.trackingRegions5244)
        crowdmisc.updateFlowTrackingRegion(displayFlowCorners, 0.5,
                                           self.trackingRegions5244)
    
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
        warped_image = cv2.warpPerspective(self.im5245, warp_matrix,
                                           self.im5245.shape[1::-1])
        warpedTrackedFrame = crowdmisc.TrackedFrame(warped_image)
        egomotion_matrix1 = np.eye(3)
        egomotion_matrix2, prev_pts2, curr_pts2 = crowd.find_egomotion(
            self.frame5244, self.frame5245,
            self.trackingRegions5244, None, None)
        egomotion_matrix3, prev_pts3, curr_pts3 = crowd.find_egomotion(
            self.frame5244, warpedTrackedFrame,
            self.trackingRegions5244, None, None)
        egomotion_matrix4, prev_pts4, curr_pts4 = crowd.find_egomotion(
            self.frame5245, warpedTrackedFrame,
            self.trackingRegions5244, None, None)
        #print np.array(egomotion_matrix3)
        #print np.dot(warp_matrix, np.array(egomotion_matrix2))
        #print np.array(warp_matrix)
        #print np.array(egomotion_matrix4)
        
        speed_no_correction = crowd.compute_horizontal_speed(
            self.frame5244, self.frame5245,
            egomotion_matrix1,
            self.trackingRegions5244)
        speed_natural_correction = crowd.compute_horizontal_speed(
            self.frame5244, self.frame5245,
            egomotion_matrix2,
            self.trackingRegions5244)
        speed_warped_image = crowd.compute_horizontal_speed(
            self.frame5244, warpedTrackedFrame,
            egomotion_matrix3,
            self.trackingRegions5244)                                                
        
        # The real displacement between the two is about ten pixels.
        # I don't know if this result is plausible.
        self.assertAlmostEqual(speed_no_correction[0], 0.006977830452)
        self.assertLess(
            (speed_natural_correction[0] - speed_warped_image[0])/speed_natural_correction[0],
            0.02)
    
    def test_compute_pedestrian_flow(self):
        base_image = cv2.imread('test_images/DSC_5156.JPG',
                               cv2.CV_LOAD_IMAGE_GRAYSCALE)
        displayFlowCorners = [(267, 449), (266, 205), (411, 198), (544, 450)]
        displayStableCorners =\
            [(5, 269), (6, 3), (231, 5), (233, 309),\
             (1058, 312), (670, 74), (684, 7), (1065, 8), 
             (62, 628), (9, 302), (164, 325), (153, 489)]
        baseTrackingRegions = crowdmisc.constructTrackingRegions(base_image)
        crowdmisc.updateEgomotionTrackingRegion(displayStableCorners, 0.5,
                                                baseTrackingRegions)
        crowdmisc.updateFlowTrackingRegion(displayFlowCorners, 0.5,
                                           baseTrackingRegions)
       
        # Test density computation
        (curr_count, prev_features, curr_features,
         velocity, velocity_inlier_idx, currTrackingRegions, 
         prev_egomotion_matrix, curr_egomotion_matrix) =\
            crowd.compute_pedestrian_flow(self.frame5244,
                                          self.frame5245,
                                          baseTrackingRegions,
                                          10, 100)

        self.assertAlmostEqual(curr_count, 100.0930081143904)
        self.assertAlmostEqual(velocity, 0.0093008114395)

        # Test tracking regions transformation.
        (curr_count, prev_features, curr_features,
        velocity, velocity_inlier_idx, currTrackingRegions,
        prev_egomotion_matrix, curr_egomotion_matrix) =\
            crowd.compute_pedestrian_flow(self.frame5817,
                                          self.frame5818,
                                          baseTrackingRegions,
                                          10, 100)
       # I eyeballed the following results to make sure they are plausible.
        self.assertListEqual(currTrackingRegions['flowCorners'],
                             [(232, 886), (229, 390), (525, 377), (795, 886)])
        self.assertListEqual(
            currTrackingRegions['stableCorners'],
            [(-313, 517), (-312, -29), (157, -18), (162, 601),
             (1809, 611), (1046, 133), (1073, -1), (1820, 12),
             (-192, 1254), (-305, 585), (19, 633), (-3, 968)] )
    
    @unittest.skip(None)
    def time_featureFinding(self):
        num_iter = 5
        start_time = time()
        for i in range(num_iter):
            cv2.goodFeaturesToTrack(self.im5817, 100, 0.5, 30,
                                    mask = self.trackingRegions['stableRegionMask'])
        end_time = time()
        print "100, mask: {}".format((end_time - start_time)/num_iter)
        start_time = time()
        for i in range(num_iter):
            cv2.goodFeaturesToTrack(self.im5817, 300, 0.5, 30,
                                    mask = self.trackingRegions['stableRegionMask'])
        end_time = time()
        print "300, mask: {}".format((end_time - start_time)/num_iter)    
        start_time = time()
        for i in range(num_iter):
            cv2.goodFeaturesToTrack(self.im5817, 100, 0.5, 30)
        end_time = time()
        print "100, no mask: {}".format((end_time - start_time)/num_iter)
        start_time = time()
        for i in range(num_iter):
            cv2.goodFeaturesToTrack(self.im5817, 300, 0.5, 30)
        end_time = time()
        print "300, no mask: {}".format((end_time - start_time)/num_iter)

def main_simulation_key_iterator():
    import main
    fx = fy = 1
    displayFlowCorners = [(267, 449), (266, 205), (411, 198), (544, 450)]
    displayFlowCorners = [(int(fx * x[0]), int(fy * x[1])) for x in displayFlowCorners]
    for point in displayFlowCorners:
        main.on_mouse(cv2.EVENT_LBUTTONDOWN, point[0], point[1], None, None)
        yield ord(' ')
    
    yield ord('e')
    displayStableCorners =\
    [(5, 269), (6, 3), (231, 5), (233, 309),\
     (1058, 312), (670, 74), (684, 7), (1065, 8), 
     (62, 628), (9, 302), (164, 325), (153, 489)]
    displayStableCorners = [(int(fx * x[0]), int(fy * x[1])) for x in displayStableCorners]
    for point in displayStableCorners:
        main.on_mouse(cv2.EVENT_LBUTTONDOWN, point[0], point[1], None, None)
        yield ord(' ')

    for i in range(1, 50):
        yield 82
        for i in range(10):
            yield ord(' ')

    yield ord('q')
    
def simulate_main():
    import main
    from time import time
    key_iterator = main_simulation_key_iterator()
    def waitKey(delayTime):
        return key_iterator.next()
    
    crowd.velocities_log = []
    start_time = time()
    try:
        main.main('/home/anatoliy/crowd/Jun12/',
                  '/home/anatoliy/crowd/recycled/',
                  '/home/anatoliy/Jun12_result/im',
                  waitKey)
    except:
        cv2.destroyAllWindows()
        raise
    end_time = time()
    
    print "Total execution time:  {0: d} seconds.".format(int(end_time - start_time))
    return crowd.velocities_log
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
#    speedtest = unittest.TestSuite()
#    speedtest.addTests([TestVision('time_featureFinding')])
#    unittest.TextTestRunner().run(speedtest)
    
    #unittest.main()

    simulate_main()
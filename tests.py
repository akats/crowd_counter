# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:02:53 2012

@author: anatoliy
"""

import cv
import numpy as np

import unittest
import numpy.testing as npt

import crowd
import crowdmisc
import draw

from crowdmisc import mode

class TestTrackingRegions(unittest.TestCase):
    def setUp(self):
        crowdmisc.mode = 'top'
                
    def test_updateFlowTrackingRegion(self):
        mask = cv.CreateMat(10, 10, cv.CV_8UC1)
        cv.SetZero(mask)
        trackingRegions = crowdmisc.constructTrackingRegions(mask)
        trackingRegions['flowMask'] = mask
        
        crowdmisc.updateFlowTrackingRegion([(2,1)], 0.5, trackingRegions)
        crowdmisc.updateFlowTrackingRegion([(2,3)], 0.5, trackingRegions)
        self.assertEquals(trackingRegions['flowCorners'], [(4,2), (4, 6)])
        crowdmisc.updateFlowTrackingRegion([(3,3),(3,1)], 0.5, trackingRegions)
        self.assertEquals(trackingRegions['flowCorners'], [(4,2), (4, 6), (6,6), (6,2)])

        trueMask = np.zeros((10, 10), np.uint8)
        trueMask[2:7][:, 4:7] = 255
        npt.assert_equal(trackingRegions['flowMask'], trueMask)
        
        crowdmisc.updateFlowTrackingRegion([(2, 2)], 0.5, trackingRegions)
        self.assertEqual(trackingRegions['flowCorners'], [(4,4)])
        self.assertEqual(trackingRegions['displayFlowCorners'], [(2,2)])
        
    def test_updateEgomotionTrackingRegion(self):
        mask = cv.CreateMat(20, 10, cv.CV_8UC1)
        cv.SetZero(mask)
        trackingRegions = crowdmisc.constructTrackingRegions(mask)
        
        crowdmisc.updateEgomotionTrackingRegion([(2,1), (2,3)], 0.5, trackingRegions)
        self.assertEquals(trackingRegions['stableCorners'], [(4,2), (4, 6)])
        crowdmisc.updateEgomotionTrackingRegion([(3,3)], 0.5, trackingRegions)
        crowdmisc.updateEgomotionTrackingRegion([(3,1)], 0.5, trackingRegions)
        self.assertEquals(trackingRegions['stableCorners'], [(4,2), (4, 6), (6,6), (6,2)])

        trueMask = np.zeros((20, 10), np.uint8)
        trueMask[2:7][:, 4:7] = 255
        npt.assert_equal(trackingRegions['stableRegionMask'], trueMask)
        
        crowdmisc.updateEgomotionTrackingRegion([(5,1)], 0.5, trackingRegions)
        crowdmisc.updateEgomotionTrackingRegion([(5,3)], 0.5, trackingRegions)
        self.assertEquals(trackingRegions['stableCorners'],\
                          [(4,2), (4, 6), (6,6), (6,2), (10,2), (10, 6)])
        crowdmisc.updateEgomotionTrackingRegion([(6,3)], 0.5, trackingRegions)
        crowdmisc.updateEgomotionTrackingRegion([(6,1)], 0.5, trackingRegions)
        self.assertEquals(trackingRegions['stableCorners'],
                          [(4,2), (4, 6), (6,6), (6,2), (10,2), (10, 6), (12,6), (12,2)])
        
        trueMask[2:7][:, 10:13] = 255
        #trueMask[0, 0] = 1
        npt.assert_equal(trackingRegions['stableRegionMask'], trueMask)
        
    def test_warpTrackingRegion(self):
        mask = cv.CreateMat(10, 10, cv.CV_8UC1)
        cv.SetZero(mask)
        trackingRegions = crowdmisc.constructTrackingRegions(mask)
        crowdmisc.updateEgomotionTrackingRegion([(2,3), (2,1), (3,1), (3,3)],
                                                 0.5, trackingRegions)
        crowdmisc.updateFlowTrackingRegion([(2,3), (2,1), (3,1), (3,3)],
                                           0.5, trackingRegions)
        warp_matrix = np.eye(3)
        warp_matrix[0, 2] = 2
        warp_matrix[1, 2] = -1
        warp_matrix = cv.fromarray(warp_matrix)
        
        out = crowdmisc.warpTrackingRegions(trackingRegions, mask, warp_matrix)
        self.assertListEqual(out['flowCorners'], [(6, 5), (6, 1), (8, 1), (8,5)])
        self.assertListEqual(out['stableCorners'], [(6, 5), (6, 1), (8, 1), (8,5)])
        self.assertAlmostEqual(out['configImageZoom'], 0.5)
        self.assertListEqual(out['displayFlowCorners'], 
                             [(3, 3), (3, 1), (4, 1), (4,3)])
        self.assertListEqual(out['displayStableCorners'], 
                             [(3, 3), (3, 1), (4, 1), (4,3)])
         # Check that trackingRegions are left unchanged
        reserveTrackingRegions = crowdmisc.constructTrackingRegions(mask)                                                 
        crowdmisc.updateEgomotionTrackingRegion([(2,3), (2,1), (3,1), (3,3)],
                                                 0.5, reserveTrackingRegions)
        crowdmisc.updateFlowTrackingRegion([(2,3), (2,1), (3,1), (3,3)],
                                           0.5, reserveTrackingRegions)
        keys = trackingRegions.keys()
        for key in keys:
            npt.assert_array_equal(np.array(reserveTrackingRegions[key]),
                                   np.array(trackingRegions[key]))
        
        true_mask = np.zeros((10, 10), dtype=np.uint8)
        true_mask[1:6][:, 6:9] = 255
        npt.assert_equal(out['flowMask'], true_mask)
        npt.assert_equal(out['stableRegionMask'], true_mask)
        
        true_scaling = np.eye(3, dtype = np.float32)
        true_scaling[(0, 1), (0, 1)] = [0.5, 0.25]
        true_shift = np.eye(3, dtype = np.float32)
        true_shift[(0, 1), (2,2)] = [-3, -0.25]
        npt.assert_equal(out['flowWarpMatrix'], np.dot(true_shift, true_scaling))

class TestTransformations(unittest.TestCase):
    def testCompute_unit_square_transform(self):
        pts = [(0,2), (0, 0), (2,0), (2,2)]
        mat = np.eye(3)
        mat[0, 0] = mat[1,1] = 0.5
        out = crowdmisc.computeUnitSquareTransform(pts)
        out = np.array(out)
        npt.assert_almost_equal(mat, out)

    def testBackproject_sample_rect(self):
        # create a warp matrix for a 10 x 20 square with an offset of (15, 25)
        warp_matrix = cv.fromarray(np.array([[0.1, 0, -1.5],
                                             [0, 0.05, -1.25],
                                             [0,0,1]], np.float32))
        sample_region = draw.backproject_sample_rect(warp_matrix, (0.5, 0.5))
        self.assertListEqual(
            sample_region,
            [(18, 40), (18,30), (22, 30), (22, 40)])
        sample_region = draw.backproject_sample_rect(warp_matrix, (0.5, 0.25))
        self.assertListEqual(
            sample_region,
            [(18, 37), (18,32), (22, 32), (22, 37)])
    
    def testCompute_sample_image_coordinates(self):

        # Test 1:  cropping on the top left
        sample_region = [(25, 25), (15, 15), (5, 25), (15, 34)]
        neighborhood_roi, local_sample_bounds =\
            draw.compute_sample_image_coordinates((100, 100), sample_region)
        # Expanding the neighborhood, we get:
        # x = 5 - 1.5 * 21 = 5 - 31.5 = -26.5
        # y = 15 - 1.5 * 20 = -15
        # width = 84
        # height = 80
        # cropping to (0, 0), we get width = 84 - 26.5 = 57.5
        # height = 80 - 15 = 65
        self.assertEqual(neighborhood_roi, 
                         (0, 0, 57.5, 65.0))
        self.assertListEqual(local_sample_bounds,
                             [(25, 25), (15, 15), (5, 25), (15, 34)])
if __name__ == '__main__':
    unittest.main()
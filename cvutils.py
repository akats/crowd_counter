# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:41:11 2012

@author: anatoliy
"""
import numpy as np
import scipy.linalg as linalg
import cv

def array2point_list(points_array):
    if type(points_array) is np.ndarray:
        points_array = points_array.reshape(-1, 2)
    points = []
    for point in points_array:
        points.append(tuple(np.round(point).astype(np.int32)))
    return points
    
def rect_intersection(rect1, rect2):
    box1 = (rect1[0], rect1[1], rect1[0]+rect1[2], rect1[1]+rect1[3])
    box2 = (rect2[0], rect2[1], rect2[0]+rect2[2], rect2[1]+rect2[3])
    intersection = (max(box1[0], box2[0]),
                    max(box1[1], box2[1]),
                    min(box1[2], box2[2]),
                    min(box1[3], box2[3]))
    return (
        intersection[0],
        intersection[1],
        intersection[2] - intersection[0],
        intersection[3] - intersection[1])

def line_intersection(pt11, pt12, pt21, pt22):
    pt11 = np.array(pt11)
    pt12 = np.array(pt12)
    pt21 = np.array(pt21)
    pt22 = np.array(pt22)
    
    vec1 = pt12 - pt11
    vec2 = pt22 - pt21
    normal1 = np.array((-vec1[1], vec1[0]))
    normal2 = np.array((-vec2[1], vec2[0]))
    intersection = linalg.solve(np.vstack((normal1, normal2)),
                                np.array((np.dot(normal1, pt11), 
                                          np.dot(normal2, pt21))))

    return tuple(intersection)
    
def line_rectangle_intersection(pt11, pt12, rect):
    corners=list()
    corners.append((rect[0], rect[1]))
    corners.append((rect[0]+rect[2], rect[1]))
    corners.append((rect[0]+rect[2], rect[1]+rect[3]))
    corners.append((rect[0], rect[1]+rect[3]))
    corners_np = np.array(corners)
    
    
    points = list()
    for line in zip(corners, corners[1:] + corners[:1]):
        intersection_point = line_intersection(pt11, pt12, line[0], line[1])
        if cv.PointPolygonTest(corners_np, intersection_point, 0) >= 0:
            points.append(intersection_point)
    
    assert(len(points)==2)
    return points[0], points[1]
    
def test_line_rectangle_intersection():
    rectangle = (0, 0, 100, 100)
    pt11 = (50, 10)
    pt12 = (40, 110)
    print line_rectangle_intersection(pt11, pt12, rectangle)
    # true intersection points:  (41, 100), (51, 0)
    
    # Return the boundary points into the order 
# bottom-right, top-right, top-left, bottom-left
def reorder_boundary_points(pts):
    I=np.argsort(pts[:,  0])
    pts = pts[I, :]
    if pts[0, 1] < pts[1,1]:
        pts = pts[[1,0, 2,3], :]
    if pts[3, 1] < pts[2,1]:
        pts = pts[[0, 1, 3,2], :]
    
    return pts

def test_reorder_boundary_points():
    boundary_pts = np.array(\
    [[ 152.,  788.],
    [ 460. , 120.],
    [ 988.  ,124.],
    [ 752.  ,904.]]
    )
    boundary_pts = boundary_pts.reshape((-1, 2));
    
    reshaped_pts = boundary_pts[[1,2,3,0], :]
    assert(np.all(boundary_pts == reorder_boundary_points(reshaped_pts)))
    reshaped_pts = boundary_pts[[0,3,1,2], :]
    assert(np.all(boundary_pts == reorder_boundary_points(reshaped_pts)))
    reshaped_pts = boundary_pts[[3,1,2,0], :]
    assert(np.all(boundary_pts == reorder_boundary_points(reshaped_pts)))
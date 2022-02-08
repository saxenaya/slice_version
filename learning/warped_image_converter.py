import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
import argparse
from cv_bridge import CvBridge

input_points = np.array([(369, 696), (971, 687), (835, 570), (478, 573)])
output_points = np.array([(-0.5, -1.5), (0.5, -1.5), (0.5, -2.5), (-0.5, -2.5)])

SCALING = (100, 100)
CENTER = (640, 1024)

transformed_output_points = []
for i in range(len(output_points)):
  transformed = np.multiply(output_points[i], SCALING) + CENTER
  transformed_output_points.append(transformed)
transformed_output_points = np.array(transformed_output_points)

class ImageConverter():
  def __init__(self, intrinsic_calib, extrinsic_calib):
    fs = cv2.FileStorage(intrinsic_calib, cv2.FILE_STORAGE_READ)
    self.K1 = fs.getNode("K1").mat()
    self.D1 = fs.getNode("D1").mat()
    self.bridge  = CvBridge()
    self.hom, _ = cv2.findHomography(input_points, transformed_output_points)

  def convert_image(self, image_msg):
    np_arr = np.fromstring(image_msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    undistorted = cv2.undistort(image_np, self.K1, self.D1)
    shape =  undistorted.shape
    warped = cv2.warpPerspective(undistorted, self.hom, shape[:-1])
    # cv2.imshow("warped", warped)
    return warped

  def convert_relative_loc(self, loc):
    transformed = np.multiply(loc, SCALING) + CENTER
    return transformed
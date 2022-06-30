#!/usr/bin/env python

import rospy
import cv2, cv_bridge
import numpy as np
from sensor_msgs.msg import Image

color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}

class ColorDetector():

    def __init__(self):
        
        rospy.init_node("color_detector")
        rospy.loginfo("Starting color_detector node")
        rgb_sub = rospy.Subscriber("/rgb/image_raw", Image, self.callback)
        self.cv_bridge = cv_bridge.CvBridge()
        self.result_pub = rospy.Publisher("/color_detect", Image, queue_size=1)

        # self.lower_bound = np.array([155, 25, 0]) # hsv
        # self.upper_bound = np.array([179, 255, 255]) # hsv
        self.lower_bound = np.array([0, 120, 200])
        self.upper_bound = np.array([18, 255, 255])

    def callback(self, rgb):
        
        # convert ros imgmsg to opencv img
        rgb = self.cv_bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        
        # red region detection 
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(rgb, self.lower_bound, self.upper_bound)
        result = cv2.bitwise_and(rgb, rgb, mask=mask)
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

        # convert opencv img to ros imgmsg
        img_msg = self.cv_bridge.cv2_to_imgmsg(result, encoding='bgr8')
        
        # publish it as topic
        self.result_pub.publish(img_msg)
        rospy.loginfo_once("Published the result as topic. check /color_detect")

    
if __name__ == '__main__':

    color_detector = ColorDetector()
    rospy.spin()
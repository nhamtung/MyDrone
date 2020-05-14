#!/usr/bin/env python
# http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

# RUN:
# $roscore
# $source devel/setup.bash
# $rosrun water_detection_ws classifierWater.py

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


def callback(msg):
    print("Received an image!")
    try:
      cv_image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='passthrough')

    except CvBridgeError as e:
      print(e)

    # rows,cols,channels = cv_image.shape
    
def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener')
    print("create node: listener")

    rospy.Subscriber("chatter_topic", Image, callback)
    print("Subscriber to chatter_topic")

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    # dataSub = UInt8MultiArray()
    listener()



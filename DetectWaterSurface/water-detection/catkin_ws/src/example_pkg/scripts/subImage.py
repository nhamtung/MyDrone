#!/usr/bin/env python
# http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber

# RUN:
# $roscore
# $source devel/setup.bash
# $rosrun example_pkg subImage.py

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


def callback(msg):
    print("Subscriber to chatter_topic!")
    try:        
        np_arr = np.fromstring(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print("INFO -- image_np.shape = " + str(image_np.shape))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        print("INFO -- image_np.shape = " + str(image_np.shape))
        # cv2.imshow("test", image_np)
        # cv2.waitKey(0)
    except CvBridgeError as e:
      print(e)
    
def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener')
    print("create node: listener")

    rospy.Subscriber("chatter_topic", CompressedImage, callback)
    print("Subscriber to chatter_topic")

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()


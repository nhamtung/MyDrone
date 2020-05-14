#!/usr/bin/env python
# http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber

# RUN:
# $roscore
# $source devel/setup.bash
# $rosrun example_pkg pubImage.py

import rospy
from std_msgs.msg import String
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

def main():
	rospy.init_node('talker', anonymous=True)
	pub = rospy.Publisher("chatter_topic", CompressedImage)
	rate = rospy.Rate(1) # 1 Hz

	while not rospy.is_shutdown():
		maskArr = None
		maskArr = cv2.imread('/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/example_pkg/data/image.png', 0)
		print("INFO -- maskArr.shape = " + str(maskArr.shape))

		msg = CompressedImage() 
		msg.header.stamp = rospy.Time.now()
		msg.format = "png"
		msg.data = np.array(cv2.imencode('.png', maskArr)[1]).tostring()
		# cv2.imshow("test", msg_maskArr)
		# cv2.waitKey(0)

		if maskArr is not None:
			pub.publish(msg)
			rospy.loginfo("INFO - published to chatter_topic")
		rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

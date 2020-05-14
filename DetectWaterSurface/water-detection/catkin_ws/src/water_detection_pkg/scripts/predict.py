#!/usr/bin/env python
# license removed for brevity

# RUN:
# $roscore
# $source devel/setup.bash
# $rosrun water_detection_ws predict.py

import rospy
from std_msgs.msg import String
import  predictFunctions
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage

# predictFunctions.moduleE("/data/beehive/ForTesting/videos/water/ocean/ocean_016.avi","/data/beehive/VideoWaterDatabase/masks/water/ocean/ocean_016.avi.png","/data/beehive/Results/",200,2,0,5,10,100,5)

VIDEO_NAME = [
  "MavicProPrecisionLanding2.mkv",    	#0
  "LandingInWater2_50fps.mkv", 			#1
  "canal.mkv"           				#2
  ]   
MODEL_NAME = [
  "tree_currentBest.pkl",           #0 - test with THRESHOLD = 0.62 (40020420035001074)
  "tree50020420070001074.pkl"       #1 - test good with THRESHOLD = 0.2 
]              
# FOLDER_VIDEO = "/content/drive/My Drive/water-detection/detectPuddle/data/"
# FOLDER_MODEL = "/content/drive/My Drive/water-detection/detectPuddle/models/"
FOLDER_VIDEO = "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/water_detection_ws/scripts/data/"
FOLDER_MODEL = "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/water_detection_ws/scripts/models/"
DIR_MASK = 0
# DIR_MASK = "/content/drive/My Drive/water-detection/detectPuddle/data/pond_mask.png"

FRAME_EACH_BLOCK = 200
DFACTOR = 2
DENSITY_MODE = 0
BOX_SIZE = 4
PATCH_SIZE = 10
NUM_FRAME_AVG = 74

PIX_HEIGHT = 250
THRESHOLD = 0.2  # water <= THRESHOLD

def main():
  dirVideo = FOLDER_VIDEO + VIDEO_NAME[1]
  print("INFO - dirVideo: " + dirVideo)
  dirModel = FOLDER_MODEL + MODEL_NAME[1]
  print("INFO - dirModel: " + dirModel)

  numFrames, height, width, fps = predictFunctions.GetFrameVideo(dirVideo)
  dFactor = height//PIX_HEIGHT
  print("INFO_dFactor = " + str(dFactor))

  rospy.init_node('talker', anonymous=True)
  pub = rospy.Publisher("chatter_topic", CompressedImage)
  rate = rospy.Rate(1) # 3 Hz

  while not rospy.is_shutdown():
  	maskArr = None
  	maskArr = cv2.imread('/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/water_detection_pkg/scripts/data/analyse/NotLand1.png', 0)
  	# maskArr = predictFunctions.Predict(dirVideo, DIR_MASK, dirModel, FRAME_EACH_BLOCK, dFactor, DENSITY_MODE, BOX_SIZE, PATCH_SIZE, NUM_FRAME_AVG, THRESHOLD)
  	print("INFO -- maskArr.shape = " + str(maskArr.shape))

  	msg = CompressedImage() 
  	msg.header.stamp = rospy.Time.now()
  	msg.format = "png"
  	msg.data = np.array(cv2.imencode('.png', maskArr)[1]).tostring()
  	# cv2.imshow("test", msg_maskArr)
  	# cv2.waitKey(0)

  	if maskArr is not None:
  		pub.publish(msg)
  		rospy.loginfo("INFO - published")
  	rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

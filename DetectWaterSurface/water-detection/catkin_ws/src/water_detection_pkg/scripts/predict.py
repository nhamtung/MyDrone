#!/usr/bin/env python
# license removed for brevity

# RUN:
# $roscore
# $source devel/setup.bash
# $rosrun water_detection_pkg predict.py

import rospy
from std_msgs.msg import String
import  predictFunctions
import cv2
import time
import os
from sklearn.externals import joblib
import numpy as np
from sensor_msgs.msg import CompressedImage

VIDEO_NAME = [
  "MavicProPrecisionLanding2.mkv",    	#0
  "LandingInWater2_50fps.mkv", 			#1
  "canal.mkv"           				#2
  ]   
MODEL_NAME = [
  "tree_currentBest.pkl",           #0 - test with THRESHOLD = 0.62 (40020420035001074)
  "tree50020420070001074.pkl"       #1 - test good with THRESHOLD = 0.2 
]              

FOLDER_VIDEO = "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/water_detection_pkg/scripts/data/"
FOLDER_MODEL = "/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/water_detection_pkg/scripts/models/"
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
  dirVideo = FOLDER_VIDEO + VIDEO_NAME[2]
  print("INFO - dirVideo: " + dirVideo)
  dirModel = FOLDER_MODEL + MODEL_NAME[1]
  print("INFO - dirModel: " + dirModel)

  numFrames, height, width, fps = predictFunctions.GetFrameVideo(dirVideo)
  dFactor = height//PIX_HEIGHT
  print("INFO_dFactor = " + str(dFactor))

  Predict(dirVideo, DIR_MASK, dirModel, FRAME_EACH_BLOCK, dFactor, DENSITY_MODE, BOX_SIZE, PATCH_SIZE, NUM_FRAME_AVG, THRESHOLD)

def Predict(dirVideo, dirMask, dirModel, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, threshold):
  print("INFO_vidpath: ", dirVideo)
  folderVideo, videoName = predictFunctions.GetNameFile(dirVideo)
  folderModel, modelName = predictFunctions.GetNameFile(dirModel)
  dirFolderOut = os.path.join(folderVideo, modelName)
  predictFunctions.CreateFolder(dirFolderOut)
  dirOutput = os.path.join(dirFolderOut, videoName)
  print("INFO_dirOutput: ", dirOutput)
  predictFunctions.CreateFolder(dirOutput)

  numFrames, height, width, fps = predictFunctions.GetFrameVideo(dirVideo)
  numBlock = numFrames//frameBlock
  print("INFO_frameEachBlock: ", frameBlock)
  print("INFO_numBlock: ", numBlock)

  model = joblib.load(dirModel) # load the SVM model
  print("Load model!")

  rospy.init_node('talker', anonymous=True)
  pub = rospy.Publisher("chatter_topic", CompressedImage)

  maskArr = None
  # maskArr = cv2.imread('/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/water_detection_pkg/scripts/data/analyse/NotLand1.png', 0)
  for i in range(numBlock):
    maskArr = PredictEachBlock(dirVideo, dirMask, dirOutput, model, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, i, threshold)
    PubImage(pub, videoName, i, maskArr)
    rospy.loginfo("INFO - published")

def PredictEachBlock(dirVideo, dirMask, dirOutput, model, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, block, threshold):
  start_time = time.time()
  mask, trueMask = predictFunctions.TestBlockVideo(dirVideo, dirMask, dirOutput, model, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, block, threshold)
  timeBlock = time.time() - start_time
  print("Time to predict each block = ", timeBlock)
  if mask is None:
    print("didn't have enough frames to run this many times")
    return None
  return mask

def PubImage(pub, videoName, block, maskArr):
  print("INFO -- maskArr.shape = " + str(maskArr.shape))
  if maskArr is not None:
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = videoName + "_block" + str(block)
    msg.format = "png"
    msg.data = np.array(cv2.imencode('.png', maskArr)[1]).tostring()
    # print("INFO -- msg: " + str(msg))

    pub.publish(msg)
    # cv2.imshow("test", msg_maskArr)
    # cv2.waitKey(0)

if __name__ == '__main__':
  try:
    main()
  except rospy.ROSInterruptException:
    pass

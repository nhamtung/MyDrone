#!/usr/bin/env python
# license removed for brevity

import  predictFunctions
import cv2
import numpy as np

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

  maskArr = None
  maskArr = cv2.imread('/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/water_detection_ws/scripts/data/analyse/Land6.png', 0)
  	# maskArr = predictFunctions.Predict(dirVideo, DIR_MASK, dirModel, FRAME_EACH_BLOCK, dFactor, DENSITY_MODE, BOX_SIZE, PATCH_SIZE, NUM_FRAME_AVG, THRESHOLD)

  	# print ("INFO - massArr.dtype: " + str(maskArr.dtype))
  	# print ("INFO - maskArr: " + str(maskArr))

  if maskArr is not None:
    AnalyseImage(maskArr)

def AnalyseImage(massArr):
  HEIGHT_ANALYSE = 100
  WIDTH_ANALYSE = 200
  THRESHOLD_WATER = 0.4

  height, width = massArr.shape
  print ("INFO - maskArr.shape: " + str(massArr.shape))

  pix_height_left = int((height - HEIGHT_ANALYSE)/2)
  pix_height_right = int((height + HEIGHT_ANALYSE)/2)
  pix_width_left = int((width - WIDTH_ANALYSE)/2)
  pix_width_right = int((width + WIDTH_ANALYSE)/2)
  # print("INFO - pix_height_left = " + str(pix_height_left))
  # print("INFO - pix_height_right = " + str(pix_height_left))
  # print("INFO - pix_width_left = " + str(pix_width_left))
  # print("INFO - pix_width_right = " + str(pix_width_right))

  crop_img = massArr[pix_height_left:pix_height_right, pix_width_left:pix_width_right]
  print ("INFO - crop_img.shape: " + str(crop_img.shape))

  # ratio = SimpleAnalyse(crop_img, HEIGHT_ANALYSE, WIDTH_ANALYSE)
  # if ratio >= THRESHOLD_WATER:
  #   print("-----------------------------------> NOT LAND!")
  # else:
  #   print("-----------------------------------> LAND!")

  ratio = WeightAnalyse(crop_img, HEIGHT_ANALYSE, WIDTH_ANALYSE)
  if ratio >= THRESHOLD_WATER:
    print("-----------------------------------> NOT LAND!")
  else:
    print("-----------------------------------> LAND!")
  # cv2.imshow("test", test)
  # cv2.waitKey(0)

def SimpleAnalyse(img, height, width):
  test = np.zeros((height, width))
  flag = 0
  weight = 0
  for i in range(height):
    for j in range(width):
      if img[i,j] == 255:
        flag = 1
      else:
        flag = 0
      weight = weight + flag
  print("INFO - weight = " + str(weight))
  ratio = weight/(height*width)
  print("INFO - ratio = " + str(ratio))
  return ratio

def WeightAnalyse(img, height, width):
  test = np.zeros((height, width))
  flag = 0
  ratio = 0
  quadrant_I = Quadrant(img, height, width, 0, int(height/2), 1, width-1, int(width/2)-1, -1)
  quadrant_II = Quadrant(img, height, width, 0, int(height/2), 1, 0, int(width/2), 1)
  quadrant_III = Quadrant(img, height, width, height-1, int(height/2)-1, -1, 0, int(width/2), 1)
  quadrant_IV = Quadrant(img, height, width, height-1, int(height/2)-1, -1, width-1, int(width/2)-1, -1)

  ratio = quadrant_I + quadrant_II + quadrant_III + quadrant_IV
  ratio = ratio/(height*width)
  print("INFO - ratio = " + str(ratio))
  return ratio

def Quadrant(img, height, width, startHeight, endHeight, stepHeight, startWidth, endWidth, stepWidth):
  weight = 0
  wei1 = 0
  wei2 = 0
  for i in range(startHeight, endHeight, stepHeight):
    wei1 = (2*i)/height
    for j in range(startWidth, endWidth, stepWidth):
      wei2 = (2*j)/width
      # print(str(i) + "    " + str(j))
      if img[i,j] == 255:
        flag = wei1*wei2
        # flag = 1
      else:
        flag = 0
      weight = weight + flag 
  # print("INFO - weight = " + str(weight))
  return weight

if __name__ == '__main__':
  main()

# http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

import numpy as np
import cv2
import os


def SaveToFile(text):
  text_file = open("/home/nhamtung/TungNV/MyDrone/DetectWaterSurface/water-detection/catkin_ws/src/water_detection_pkg/scripts/data/result.txt", "a")
  text_file.write(text + "\n")
  text_file.close()
  print("INFO - Saved result to file ----------> " + text)

def AnalyseImage(msg, massArr):
  HEIGHT_ANALYSE = 100
  WIDTH_ANALYSE = 200
  THRESHOLD_WATER = 0.3

  print ("INFO - maskArr.shape: " + str(massArr.shape))
  height, width = massArr.shape

  pix_height_left = int((height - HEIGHT_ANALYSE)/2)
  pix_height_right = int((height + HEIGHT_ANALYSE)/2)
  pix_width_left = int((width - WIDTH_ANALYSE)/2)
  pix_width_right = int((width + WIDTH_ANALYSE)/2)
  # print("INFO - pix_height_left = " + str(pix_height_left))
  # print("INFO - pix_height_right = " + str(pix_height_left))
  # print("INFO - pix_width_left = " + str(pix_width_left))
  # print("INFO - pix_width_right = " + str(pix_width_right))

  crop_img = massArr[pix_height_left:pix_height_right, pix_width_left:pix_width_right, ]
  print ("INFO - crop_img.shape: " + str(crop_img.shape))

  # ratio = SimpleAnalyse(crop_img, HEIGHT_ANALYSE, WIDTH_ANALYSE)
  # if ratio >= THRESHOLD_WATER:
  #   print("-----------------------------------> NOT LAND!")
  # else:
  #   print("-----------------------------------> LAND!")

  ratio = WeightAnalyse(crop_img, HEIGHT_ANALYSE, WIDTH_ANALYSE)
  if ratio >= THRESHOLD_WATER:
    textSave = msg.header.frame_id + ": NOT LAND ----- " + str(ratio) 
    SaveToFile(textSave)
    return "NOT_LAND"
  else:
    textSave = msg.header.frame_id + ": LAND --------- " + str(ratio)
    SaveToFile(textSave)
    return "LAND"

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
  # print("INFO - ratio = " + str(ratio))
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




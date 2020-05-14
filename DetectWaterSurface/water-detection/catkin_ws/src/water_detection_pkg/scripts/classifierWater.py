#!/usr/bin/env python
# http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

# RUN:
# $roscore
# $source devel/setup.bash
# $rosrun water_detection_pkg classifierWater.py

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


def callback(msg):
    print("Received an image!")
    try:        
      np_arr = np.fromstring(msg.data, np.uint8)
      image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
      print("INFO -- image_np.shape = " + str(image_np.shape))
      image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
      isLand = AnalyseImage(msg, image_np)
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


def AnalyseImage(msg, massArr):
  HEIGHT_ANALYSE = 100
  WIDTH_ANALYSE = 200
  THRESHOLD_WATER = 0.4

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
    print("-----------------------------------> " + msg.header.frame_id + ": NOT LAND!")
    return "NOT_LAND"
  else:
    print("-----------------------------------> " + msg.header.frame_id + " LAND!")
    return "LAND"

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
    # dataSub = UInt8MultiArray()
    listener()



import cv2
import numpy as np
import helperFunc
import os

VIDEO_FOLDER = [
  "smartphone",         #0
  "drone"				#1
]  

VIDEO_WATER = "/home/nhamtung/Documents/MyDrone/water-detection/dataset/videos/water/"
MASK_WATER = "/home/nhamtung/Documents/MyDrone/water-detection/dataset/masks/"

def GetInfoVideo(dirVideo):
  cap = cv2.VideoCapture(dirVideo)
  if (cap.isOpened() == False):
    print("Error opening video stream or file")
    return 0
  numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  print("INFO_numFrames: ", numFrames)
  print("INFO_height: ", height)
  print("INFO_width: ", width)
  fps = int(cap.get(5))
  print("INFO_fps: ", fps)
  return numFrames, height, width, fps

def CreateMaskWater(height, width, mask):
    img = np.zeros((height, width, 1), dtype=np.uint8)
    for i in range(0, height, 1):
    	for j in range(0, width, 1):
    		if mask == "Water":
    			img[i,j] = 255
    return img

def CreateFolder (dirFolder):
  if not os.path.exists(dirFolder):
    os.mkdir(dirFolder)
    print("Directory ", dirFolder, " Created")
  else:
    print("Directory ", dirFolder, " already exists")
  return dirFolder

def GetNameFile(dirPath):
	folderName, nameFile = os.path.split(dirPath)
	nameFile = nameFile[:-4]
	print("INFO_folderName: ", folderName)
	print("INFO_nameFile: ", nameFile)
	return folderName, nameFile

def SaveImage(dirOutput, nameImage, image):
  dirImage = os.path.join(dirOutput, nameImage) + '.png'
  cv2.imwrite(dirImage, image)
  print("Saved " + dirImage + ": " + dirImage)

def main():
	nameFolder = VIDEO_FOLDER[0]
	dirVideoFolder = VIDEO_WATER + nameFolder
	print("INFO - dirVideoFolder: " + dirVideoFolder)
	for vids in os.listdir(dirVideoFolder):
		print("INFO - Video Water: ", vids)
		dirVideo = dirVideoFolder + "/" + vids
		print("INFO - dirVideo: " + dirVideo)
		numFrames, height, width, fps = GetInfoVideo(dirVideo)
		imgMask = CreateMaskWater(height, width, "Water")
		dirMaskFolder = MASK_WATER + nameFolder
		CreateFolder(dirMaskFolder)
		folderName, nameFile = GetNameFile(dirVideo)
		SaveImage(dirMaskFolder, nameFile, imgMask)

if __name__ == '__main__':
  main()
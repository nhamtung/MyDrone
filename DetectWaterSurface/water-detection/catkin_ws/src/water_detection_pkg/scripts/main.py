import  predictFunctions
import cv2

# predictFunctions.moduleE("/data/beehive/ForTesting/videos/water/ocean/ocean_016.avi","/data/beehive/VideoWaterDatabase/masks/water/ocean/ocean_016.avi.png","/data/beehive/Results/",200,2,0,5,10,100,5)

VIDEO_NAME = [
  "AutoPrecisionLand1080_2.mkv",    #0
  "canal.mkv",                      #1
  "DjiDroneAutoland1080.mkv",       #2
  "DjiMavicProFloatKit3_30fps.mkv", #3
  "DjiMavicProFloatKit3_50fps.mkv", #4
  "DjiMavicProFloatKit5_30fps.mkv", #5
  "DjiMavicProFloatKit5_50fps.mkv", #6
  "DjiMavicProFloatKit7_30fps.mkv", #7
  "DjiMavicProFloatKit7_50fps.mkv", #8
  "FlyingOverWater.mkv",            #9
  "LandingInWater1_30fps.mkv",      #10
  "LandingInWater1_50fps.mkv",      #11
  "LandingInWater2_30fps.mkv",      #12
  "LandingInWater2_50fps.mkv",      #13
  "M210DjiDroneLanding3.mkv",       #14
  "MavicProPrecisionLanding.mkv",   #15
  "MavicProPrecisionLanding2.mkv",  #16
  "MavicProPrecisionLanding3.mkv",  #17
  "NoneWater_30fps.mkv",            #18
  "NoneWater_50fps.mkv",            #19
  "pond_30fps.mkv",                 #20
  "pond_50fps.mkv",                 #21
  "canal_023.avi"                   #22
  ]   
MODEL_NAME = [
  "tree_currentBest.pkl",           #0 - test with THRESHOLD = 0.62 (40020420035001074)
  "model_tree.pkl",                 #1 - test with THRESHOLD = 0.3 (40020420035001074)
  "tree40020420035001050.pkl",      #2
  "tree40020420035001074.pkl",      #3
  "tree50020420070001074.pkl",      #4 - test good with THRESHOLD = 0.2 
  "tree50020410070001050.pkl",      #5 - test with THRESHOLD = 0.25 
  "tree5002045070001050.pkl"        #6 - test with THRESHOLD = 0.2 
]              
# FOLDER_VIDEO = "/content/drive/My Drive/water-detection/detectPuddle/data/"
# FOLDER_MODEL = "/content/drive/My Drive/water-detection/detectPuddle/models/"
FOLDER_VIDEO = "./data/"
FOLDER_MODEL = "./models/"
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
  dirVideo = FOLDER_VIDEO + VIDEO_NAME[13]
  print("INFO - dirVideo: ", dirVideo)
  dirModel = FOLDER_MODEL + MODEL_NAME[4]
  print("INFO - dirModel: ", dirModel)

  numFrames, height, width, fps = predictFunctions.GetFrameVideo(dirVideo)
  dFactor = height//PIX_HEIGHT
  print("INFO_dFactor = ", dFactor)

  predictFunctions.Predict(dirVideo, DIR_MASK, dirModel, FRAME_EACH_BLOCK, dFactor, DENSITY_MODE, BOX_SIZE, PATCH_SIZE, NUM_FRAME_AVG, THRESHOLD)
  # predictFunctions.moduleE(dirVideo,DIR_MASK,"/content/drive/My Drive/water-detection/detectPuddle/data/pond_016",200,2,0,5,10,100,4)

if __name__ == '__main__':
   main()

   
import cv2
import numpy as np
import sys
import Imagetransformations
import containerFunctions as ct
import features
import helperFunc
from sklearn.externals import joblib
import time
import os

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

def Predict(dirVideo, dirMask, dirModel, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, threshold):
    print("INFO_vidpath: ", dirVideo)
    folderVideo, videoName = GetNameFile(dirVideo)
    folderModel, modelName = GetNameFile(dirModel)
    dirFolderOut = os.path.join(folderVideo, modelName)
    CreateFolder(dirFolderOut)
    dirOutput = os.path.join(dirFolderOut, videoName)
    print("INFO_dirOutput: ", dirOutput)
    CreateFolder(dirOutput)

    numFrames, height, width, fps = GetFrameVideo(dirVideo)
    numBlock = numFrames//frameBlock
    print("INFO_frameEachBlock: ", frameBlock)
    print("INFO_numBlock: ", numBlock)

    model = joblib.load(dirModel) # load the SVM model
    print("Load model!")

    numBlock = 1
    maskArr = None
    for i in range(numBlock):
      start_time = time.time()
      mask, trueMask = TestBlockVideo(dirVideo, dirMask, dirOutput, model, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, i, threshold)
      timeBlock = time.time() - start_time
      print("Time to predict each block = ", timeBlock)
      if mask is None:
        print("didn't have enough frames to run this many times")
        break
      if maskArr is None:
        maskArr = mask
      else:
        maskArr = np.dstack((maskArr,mask))

    if maskArr is None:
      print("video is too short, no mask generated")
      sys.exit()
    if numBlock > 1:
      finalMask = np.sum(maskArr, 2)
      logical = finalMask < (255 * 2 * numBlock/4 )
      finalMask[logical] = 0
      finalMask[finalMask != 0] = 255
      SaveImage(dirOutput, "FinalMask", finalMask)

    if trueMask is not None:
        FigureOutNumbers(finalMask, trueMask)

    # SaveImage(dirOutput, "maskArr", maskArr)
    # print ("INFO - maskArr: " + str(maskArr))
    return maskArr

def GetFrameVideo(dirVideo):
  # print("INFO - dirVideo: " + dirVideo)
  cap = cv2.VideoCapture(dirVideo)
  if (cap.isOpened() == False):
    print("Error opening video stream or file")
    return 0, 0, 0, 0
  numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  print("INFO_numFrames: ", numFrames)
  print("INFO_height: ", height)
  print("INFO_width: ", width)
  fps = int(cap.get(5))
  print("INFO_fps: ", fps)
  return numFrames, height, width, fps

def TestBlockVideo(dirVideo, dirMask, dirOutput, model, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, block, threshold):
    print("INFO_block: ", block)
    completeVid, feature, trueMask = GetFeaturesAndMask(dirVideo, dirMask, dirOutput, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, block)
    print("INFO_feature.shape: ", feature.shape)

    if feature is None:
        return None,None
    width = feature.shape[1]
    height = feature.shape[0]
    minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))

    isWaterFound = np.zeros((height,width),  dtype = np.int)
    newShape = feature.reshape((feature.shape[0] * feature.shape[1], feature.shape[2]))
    # print("TungNV_newShape: ", newShape.shape)

    prob = model.predict_proba(newShape)[:, 0]
    prob = prob.reshape((feature.shape[0], feature.shape[1]))

    probabilityMask = prob
    isWaterFound[prob <= threshold] = True
    isWaterFound[prob > threshold] = False
    isWaterFound = isWaterFound.astype(np.uint8)
    isWaterFound = isWaterFound[minrand:height-minrand, minrand:width-minrand]

    if trueMask is not None:
        trueMask = trueMask[minrand:height-minrand,minrand:width-minrand]
        trueMask[trueMask == 1] = 255
        
    isWaterFound[isWaterFound == 1] = 255
    SaveImage(dirOutput, "block" + str(block) + '_before_regularization', isWaterFound)

    probabilityMask = probabilityMask[minrand:height-minrand, minrand:width-minrand]
    #probabilityMask = (probabilityMask-np.min(probabilityMask))/(np.max(probabilityMask)-np.min(probabilityMask))
    
    start_time = time.time()
    for i in range(11):
      isWaterFound = RegularizeFrame(isWaterFound,probabilityMask,.2)
    print("Time to Regularize Frame = ", time.time() - start_time)
    #isWaterFound = cv2.morphologyEx(isWaterFound, cv2.MORPH_OPEN, kernel)

    width = isWaterFound.shape[1]
    height = isWaterFound.shape[0]
    isWaterFound = isWaterFound[11:height-11, 11:width-11]

    if trueMask is not None:
        # cv2.imshow("mask created",isWaterFound)
        trueMask = trueMask[11:height - 11, 11:width - 11]
        # cv2.imshow("old mask", trueMask)
        # cv2.waitKey(0)
        FigureOutNumbers(isWaterFound, trueMask)
    SaveImage(dirOutput, "block" + str(block) + '_newMask_direct', isWaterFound)

    maskedImg = MaskFrameWithMyMask(completeVid[11:height-11, 11:width-11, int(frameBlock/2)], isWaterFound)
    SaveImage(dirOutput, "block" + str(block) + '_Masked_frame_from_video', maskedImg)

    # print("TungNV_isWaterFound.shape: ", isWaterFound.shape)
    # print("TungNV_trueMask.shape: ", trueMask.shape)
    return isWaterFound, trueMask

def GetFeaturesAndMask(dirVideo, dirMask, dirOutput, frameBlock, dFactor, densityMode, boxSize, patchSize, numFramesAvg, block):
    frameMid, blockVid, completeVid, preprocess = ct.PreprocessVideo(dirVideo, dirOutput, frameBlock, dFactor, densityMode, block)
    if preprocess is None:
      print("INFO_preprocess.shape: ", preprocess.shape)
      return None, None

    SaveImage(dirOutput, "block" + str(block) + '_Frame_Mid', frameMid)
    # dirBlock = os.path.join(dirOutput, "grayBlock" + str(block) + ".avi")
    # helperFunc.saveVid(completeVid, dirBlock)
    # print("Saved gray video!")
    dirBlock = os.path.join(dirOutput, "block" + str(block) + ".avi")
    print("INFO - blockVid.shape: ", blockVid.shape)
    helperFunc.SaveVidColor(blockVid, dirBlock)
    print("Saved video block: ", dirBlock)
    dirBlock = os.path.join(dirOutput, "block" + str(block) + "_residual_most_recent.avi")
    helperFunc.saveVid(preprocess, dirBlock)
    print("Saved video residual most recent: ", dirBlock)

    features, isWater = GetFeatures(preprocess, dirMask, dFactor, boxSize, patchSize, numFramesAvg)
    return completeVid, features, isWater

def GetFeatures(preprocessedVid, dirMask, dscale, boxSize, patchSize, numFramesAvg):
    start_time = time.time()
    #sets up vars
    width = preprocessedVid.shape[1]
    height = preprocessedVid.shape[0]

    #water == True in mask
    if dirMask is not 0:
      print("INFO_dirMask: ", dirMask)
      mask = features.CreateMask(dirMask, dscale)
      mask = mask[:,:,0]
      print("INFO_mask.shape: ", mask.shape)
      isWater = mask.astype(np.uint8)
    else:
      isWater = None

    # print("TungNV_boxSize: ", boxSize)
    # print("TungNV_patchSize: ", patchSize)
    #size of edge around the image-- will later crop this part out
    minrand = max(int(boxSize/2 + 1), int(patchSize/2))
    # print("TungNV_minrand: ", minrand)

    #obtain temporal feature and corp the image
    temporalFeat = features.FourierTransformFullImage(preprocessedVid, boxSize)
    temporalFeat[0:minrand,:] = 0
    temporalFeat[:,0:minrand] = 0
    temporalFeat[height-minrand:height, :] = 0
    temporalFeat[:,width-minrand:width] = 0
    spaceFeat = features.SpatialFeaturesFullImage(preprocessedVid, patchSize, numFramesAvg)
    print("INFO_temporalFeat.shape: ", temporalFeat.shape)
    print("INFO_spaceFeat.shape: ", spaceFeat.shape)
    combinedFeature = np.concatenate((temporalFeat, spaceFeat), 2)

    print("predictFunctions.GetFeatures: Finished computing unified featureSpace")
    print("INFO_combinedFeature.shape: ", combinedFeature.shape)
    combinedFeature = combinedFeature.astype(np.float32)

    print("Time to get Features: ", time.time() - start_time)
    return combinedFeature, isWater

def RegularizeFrame(myMask, probabilityMask, gamma):
  newMask = np.zeros(myMask.shape)
  for i in range(1, myMask.shape[0]-1):
    for j in range(1, myMask.shape[1]-1):
      zeros = probabilityMask[i,j] + (gamma * regularizeHelper(myMask,i,j,0))
      twoFiftyFive = 1 - probabilityMask[i,j] + (gamma * regularizeHelper(myMask,i,j,255))
      if zeros < twoFiftyFive:
        newMask[i,j] = 0
      else:
        newMask[i, j] = 255
  return newMask

def FigureOutNumbers(createdMask, trueMask):
    print("percent accuracy: " + str(100*np.sum(trueMask == createdMask) / createdMask.size))
    cond1 = (createdMask != trueMask) & (trueMask == 0)
    falsePos =  createdMask[cond1]
    print("percent false positive: " + str(100*(len(falsePos)/trueMask.size)))

    cond2 = (createdMask != trueMask) & (trueMask == 255)
    falseNeg = createdMask[cond2]
    print("percent false negative: "+ str(100*(len(falseNeg)/trueMask.size)))

def MaskFrameWithMyMask(frameFromVid,ourMask):
    frameFromVid[ourMask == 0] = 0
    # cv2.imshow("windowName", frameFromVid)
    # cv2.waitKey(0)
    return frameFromVid








###############################################################################################################################













def moduleE(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,patchSize,numFramesAvg,numVids):
    maskArr = None
    for i in range(numVids):
        mask,trueMask = testFullVid(vidpath, maskpath, outputFolder, numFrames, dFactor, densityMode, boxSize, patchSize, numFramesAvg,i)
        if mask is None:
            print("didn't have enough frames to run this many times")
            break
        if maskArr is None:
            maskArr = mask
        else:
            maskArr = np.dstack((maskArr,mask))

    if maskArr is None:
        print("video is too short, no mask generated")
        sys.exit()
    finalMask = np.sum(maskArr,2)
    logical = finalMask < (255 * 2 * numVids/4 )
    finalMask[logical] = 0
    finalMask[finalMask != 0] = 255
    cv2.imshow("normalizedMask",finalMask)
    cv2.imwrite(outputFolder +"FinalMask.png",finalMask)
    cv2.waitKey(0)
    if trueMask is not None:
        FigureOutNumbers(finalMask, trueMask)


def moduleD(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,numbofFrameSearch,numbofSamples,patchSize,numFramesAvg,numVids):
    maskArr = None
    for i in range(numVids):
        mask,trueMask = testFullVid(vidpath, maskpath, outputFolder, numFrames, dFactor, densityMode, boxSize, numbofFrameSearch,
                    numbofSamples, patchSize, numFramesAvg,i)
        if mask is None:
            print("didn't have enough frames to run this many times")
            break
        if maskArr is None:
            maskArr = mask
        else:
            maskArr = np.dstack((maskArr,mask))
    if maskArr is None:
            exit(0)
    finalMask = np.sum(maskArr,2)
    logical = finalMask < (255 * numVids/2)
    finalMask[logical] = 0
    finalMask[finalMask != 0] = 255
    cv2.imshow("normalizedMask",finalMask)
    cv2.imwrite("FinalMask.png",finalMask)
    cv2.waitKey(0)
    if trueMask is not None:
        FigureOutNumbers(finalMask, trueMask)

def moduleC(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,patchSize,numFramesAvg,vidNum):
    preprocess = ct.preprocessVideo(vidpath,numFrames,dFactor,densityMode,vidNum)
    if preprocess is None:
        return None,None
    helperFunc.saveVid(preprocess, outputFolder + vidpath[-12:-4] + "residual_most_recent.avi")
    features, isWater = getFeatures(preprocess,maskpath,dFactor,boxSize,patchSize,numFramesAvg)
    return features, isWater
def getFeatures(preprocessedVid,maskpath,dscale,boxSize,patchSize,numFramesAvg):
    #sets up vars
    width = preprocessedVid.shape[1]
    height = preprocessedVid.shape[0]

    #water == True in mask
    if maskpath is not 0:
        mask = features.createMask(maskpath,dscale)
        mask = mask[:,:,0]
        isWater = mask.astype(np.uint8)
    else:
        isWater = None

    #size of edge around the image-- will later crop this part out
    minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))

    #obtain temporal feature and corp the image
    temporalFeat = features.fourierTransformFullImage(preprocessedVid,boxSize)
    temporalFeat[0:minrand,:] = 0
    temporalFeat[:,0:minrand] = 0
    temporalFeat[height-minrand:height, :] = 0
    temporalFeat[:,width-minrand:width] = 0
    spaceFeat = features.spatialFeaturesFullImage(preprocessedVid,patchSize,numFramesAvg)
    combinedFeature = np.concatenate((temporalFeat,spaceFeat),2)


    print("finished computing unified featureSpace")

    combinedFeature = combinedFeature.astype(np.float32)
    return combinedFeature, isWater

def figureOutNumbers(createdMask, trueMask):
    print("percent accuracy: " + str(100*np.sum(trueMask == createdMask) / createdMask.size))
    cond1 = (createdMask != trueMask) & (trueMask == 0)
    falsePos =  createdMask[cond1]
    print("percent false positive: " + str(100*(len(falsePos)/trueMask.size)))

    cond2 = (createdMask != trueMask) & (trueMask == 255)
    falseNeg = createdMask[cond2]
    print("percent false negative: "+ str(100*(len(falseNeg)/trueMask.size)))

def maskFrameWithMyMask(frameFromVid,ourMask):
    frameFromVid[ourMask == 0] = 0
    cv2.imshow("windowName", frameFromVid)
    cv2.waitKey(0)
    return frameFromVid

def regularizeFrame(myMask, probabilityMask,gamma):
    newMask = np.zeros(myMask.shape)
    for i in range(1,myMask.shape[0]-1):
        for j in range(1,myMask.shape[1]-1):
            zeros = probabilityMask[i,j] + (gamma * regularizeHelper(myMask,i,j,0))
            twoFiftyFive = 1 - probabilityMask[i,j] + (gamma * regularizeHelper(myMask,i,j,255))
            if zeros < twoFiftyFive:
                newMask[i,j] = 0
            else:
                newMask[i, j] = 255
    return newMask
def regularizeHelper(myMask, i, j,checkValue):
    up = checkValue != myMask[i+1,j]
    down =checkValue != myMask[i - 1, j]
    left = checkValue != myMask[i, j-1]
    right = checkValue != myMask[i, j + 1]
    topleft = checkValue != myMask[i + 1, j -1]
    topRight = checkValue != myMask[i + 1, j + 1]
    bottomLeft = checkValue != myMask[i - 1, j - 1]
    bottomRight = checkValue != myMask[i - 1, j + 1]
    sum1 = int(up) + int(down) + int(left) +int(right) +int(topleft) + int(topRight) + int(bottomLeft) + int(bottomRight)
    return sum1

def testFullVid(vidpath,maskpath,outputFolder, numFrames, dFactor, densityMode,boxSize,patchSize,numFramesAvg,vidNum):
    feature, trueMask = moduleC(vidpath,maskpath,outputFolder,numFrames,dFactor,densityMode,boxSize,patchSize,numFramesAvg,vidNum)
    if feature is None:
        return None,None
    width = feature.shape[1]
    height = feature.shape[0]
    minrand = max(int(boxSize / 2 + 1), int(patchSize / 2))
    # load the SVM model
    model = joblib.load('/content/drive/My Drive/water-detection/detectPuddle/models/tree_currentBest.pkl')
    isWaterFound = np.zeros((height,width),  dtype = np.int)
    newShape = feature.reshape((feature.shape[0] * feature.shape[1], feature.shape[2]))
    prob = model.predict_proba(newShape)[:, 0]
    prob = prob.reshape((feature.shape[0], feature.shape[1]))
    probabilityMask = prob
    isWaterFound[prob<.5] = True
    isWaterFound[prob>.5] = False
    isWaterFound = isWaterFound.astype(np.uint8)
    isWaterFound = isWaterFound[minrand:height-minrand, minrand:width-minrand]
    if trueMask is not None:
        trueMask = trueMask[minrand:height-minrand,minrand:width-minrand]
        trueMask[trueMask == 1] = 255
    isWaterFound[isWaterFound == 1] = 255
    beforeReg = outputFolder + str(vidNum) + '_before_regularization' + '.png'
    cv2.imwrite(beforeReg, isWaterFound)
    probabilityMask = probabilityMask[minrand:height-minrand, minrand:width-minrand]
    #probabilityMask = (probabilityMask-np.min(probabilityMask))/(np.max(probabilityMask)-np.min(probabilityMask))
    for i in range(11):
        isWaterFound = regularizeFrame(isWaterFound,probabilityMask,.2)
    #isWaterFound = cv2.morphologyEx(isWaterFound, cv2.MORPH_OPEN, kernel)
    width = isWaterFound.shape[1]
    height = isWaterFound.shape[0]
    isWaterFound = isWaterFound[11:height-11, 11:width-11]
    if trueMask is not None:
        cv2.imshow("mask created",isWaterFound)
        trueMask = trueMask[11:height - 11, 11:width - 11]
        cv2.imshow("old mask", trueMask)
        cv2.waitKey(0)
        FigureOutNumbers(isWaterFound, trueMask)
    cv2.imwrite(outputFolder + str(vidNum) + 'newMask_direct.png', isWaterFound)
    completeVid = Imagetransformations.importandgrayscale(vidpath,numFrames,dFactor,vidNum)
    maskedImg = maskFrameWithMyMask(completeVid[11:height-11, 11:width-11,int(numFrames/2)],isWaterFound)
    cv2.imwrite(outputFolder + str(vidNum) + 'Masked_frame_from_video.png', maskedImg)
    return isWaterFound, trueMask

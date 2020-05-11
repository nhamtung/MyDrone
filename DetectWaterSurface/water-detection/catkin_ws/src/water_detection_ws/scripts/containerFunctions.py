import os
import random
import cv2
import numpy as np
import Imagetransformations
import features
import plotFuncs
import time

def PreprocessVideo(dirVideo, dirOutput, frameBlock, dFactor, densityMode, block):
    start_time = time.time()
    #sets vars if not put in
    if ((dFactor is None) or (dFactor == -1)):
      dFactor = 1;
    if (densityMode is None):
      densityMode = 0

    frameMid, blockVid, gray = Imagetransformations.ImportAndGrayScale(dirVideo, frameBlock, dFactor, block)
    print("INFO - gray.shape: ", gray.shape)
    if gray is None:
        return None
    #helperFunc.playVid(gray,"grayVid.avi")

    if(densityMode == 1):
        modeFrame = Imagetransformations.GetDensityModeFrame(gray)
        # dirModeFrame = dirOutput + "/mode_density.png"
        # cv2.imwrite(dirModeFrame, modeFrame)
        # print("Saved image mode_density: " + dirModeFrame)
    else:
        modeFrame = Imagetransformations.GetDirectModeFrame(gray)
        # dirModeFrame = dirOutput + "/mode_Direct.png"
        # cv2.imwrite(dirModeFrame, modeFrame)
        # print("Saved image mode_Direct: ", dirModeFrame)

    residual = Imagetransformations.CreateResidual(gray, modeFrame)
    print("Time Preprocess Video: ", time.time() - start_time)
    return frameMid, blockVid, gray, residual

def GetFeaturesPoints(preprocessedVid, dirMask, dscale, boxSize, temporalLength, numbOfSamples, patchSize, numFramesAvg):
    #sets up vars
    width = preprocessedVid.shape[1]
    height = preprocessedVid.shape[0]
    numFrames = preprocessedVid.shape[2]
    random.seed(0)

    #water == True in mask
    if dirMask == 0:
        mask = np.zeros((height, width))
    else:
        mask = features.createMask(dirMask, dscale)
        mask = mask[:,:,0]

    #sets up arrays for loop
    isWater = np.zeros((1,numbOfSamples),dtype=bool)
    temporalArr = np.zeros((numbOfSamples, temporalLength))
    spatialArr = np.zeros((numbOfSamples, 256))
    minrand = max(int(boxSize/2+1),int(patchSize/2))
    totalFeatures= np.zeros((numbOfSamples,temporalLength+256))

    #gets array with amount of features
    for i in range(numbOfSamples):
        randx = random.randrange(minrand, width - minrand)
        randy = random.randrange(minrand, height - minrand)
        randz = random.randrange(int(temporalLength/2), numFrames - int(temporalLength/2))
        isWater[0,i] = mask[randy,randx]
        temporalArr[i,:] = features.FourierTransform(preprocessedVid,randx,randy,randz,boxSize,temporalLength)
        temporalFeat = temporalArr[i,:]

        spatialArr[i,:] = features.SpatialFeatures(preprocessedVid,randx,randy,randz,patchSize,numFramesAvg)
        spaceFeat = spatialArr[i,:]
        combinedFeature = np.concatenate((np.reshape(temporalFeat,(temporalFeat.size)),spaceFeat))
        totalFeatures[i,:] = combinedFeature
    #plotFuncs.PlotTemporalFeatures(temporalArr, isWater,numbOfSamples)
    #features.plotSpatialFeatures(spatialArr,isWater,numbOfSamples)

    print("finished computing unified featureSpace")
    isWater = isWater.astype(int)
    return totalFeatures, isWater

def ModuleB(dirVideo, dirMask, frameEachBlock, dFactor, densityMode, boxSize, temporalLength, numbOfSamples, patchSize, numFramesAvg):
    dirOut = ""
    frameMid, blockVid, gray, preprocess = PreprocessVideo(dirVideo, dirOut, frameEachBlock, dFactor, densityMode, 1)
    features, isWater = GetFeaturesPoints(preprocess, dirMask, dFactor, boxSize, temporalLength, numbOfSamples, patchSize, numFramesAvg)
    return features, isWater

#for training the classifier
def LoopsThroughAllVids(pathToVidsFolder, pathToMasksPondFolder, pathToOtherTextures, numFrames, dFactor, densityMode, boxSize, temporalLength, numbOfSamples, patchSize, numFramesAvg):
    #sets up vars
    totalFeatureSet = None
    isWateragg = None
    halfamountofVidsinFolder = 25
    #goes through first 20 vids
    for folders in os.listdir(pathToVidsFolder):
        counter = 0
        for vids in os.listdir(pathToVidsFolder + "/" + folders):
            print("INFO - Video Water: ", vids)
            if counter > halfamountofVidsinFolder:
                break
            nameMask = pathToMasksPondFolder + folders + "/" + vids[:-3] + 'png'
            nameVid = pathToVidsFolder + folders + "/" + vids
            #obtains features for video then concats them to matrix
            feature,isWater = ModuleB(nameVid, nameMask, numFrames, dFactor, densityMode, boxSize, temporalLength, numbOfSamples, patchSize, numFramesAvg)
            AddFeatureToArr(totalFeatureSet, isWateragg, feature, isWater)
            counter+= 1
            print(counter)
        print(folders)
    halfamountofVidsinFolder = 20
    for folders in os.listdir(pathToOtherTextures):
        counter = 0
        for vids in os.listdir(pathToOtherTextures + folders):
            print("INFO - Video Non Water: ", vids)
            if counter > halfamountofVidsinFolder:
                break
            nameVid = pathToOtherTextures + folders + "/" + vids
            # obtains features for video then concats them to matrix
            feature, isWater = ModuleB(nameVid, 0, numFrames, dFactor, densityMode, boxSize, temporalLength, numbOfSamples, patchSize, numFramesAvg)
            AddFeatureToArr(totalFeatureSet, isWateragg, feature, isWater)
            counter += 1
            print(counter)

    #turns water into correct orientation
    isWateragg = isWateragg * 1
    isWateragg = np.transpose(isWateragg)
    return totalFeatureSet, isWateragg

def AddFeatureToArr(totalFeatureSet, isWateragg, feature, isWater):
  if totalFeatureSet is None:
    totalFeatureSet = feature
  else:
    totalFeatureSet = np.concatenate((totalFeatureSet, feature), axis=0)
  if isWateragg is None:
    isWateragg = isWater
  else:
    isWateragg = np.concatenate((isWateragg, isWater), axis=1)
  return totalFeatureSet, isWateragg

def GetInfoVideo(dirVideo):
  cap = cv2.VideoCapture(dirVideo)
  if (cap.isOpened() == False):
    print("Error opening video stream or file")
    return 0
  numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  print("INFO - numFrames: ", numFrames)
  print("INFO - height: ", height)
  print("INFO - width: ", width)
  fps = int(cap.get(5))
  print("INFO - fps: ", fps)
  return numFrames, height, width, fps

##########################################################################################################################




def preprocessVideo(path,numFrames, dFactor, densityMode,vidNum):
    #capture and subtract water reflections and colours from the video frames
    if path is not 0:
        cap = cv2.VideoCapture(path)
        if(numFrames is None or numFrames == -1):
            numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if ((dFactor is None) or (dFactor == -1)):
        dFactor = 1;
    if (densityMode is None):
        densityMode = 0

    gray = Imagetransformations.importandgrayscale(path,numFrames,dFactor,vidNum)
    if gray is None:
        return None
    #helperFunc.playVid(gray,"grayVid.avi")
    if(densityMode == 1):
        modeFrame = Imagetransformations.getDensitytModeFrame(gray)
        #cv2.imwrite('mode_density.png', modeFrame)
    else:
        modeFrame = Imagetransformations.getDirectModeFrame(gray)
        cv2.imwrite('mode_Direct.png', modeFrame)
    if path is not 0:
        cap.release()
    residual = Imagetransformations.createResidual(gray,modeFrame)
    return residual

def getFeaturesPoints(preprocessedVid,maskpath,dscale,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg):
    #sets up vars
    width = preprocessedVid.shape[1]
    height = preprocessedVid.shape[0]
    numFrames = preprocessedVid.shape[2]
    random.seed(0)

    #water == True in mask
    if maskpath == 0:
        mask = np.zeros((height,width))
    else:
        mask = features.createMask(maskpath,dscale)
        mask = mask[:,:,0]

    #sets up arrays for loop
    isWater = np.zeros((1,numbofSamples),dtype=bool)
    temporalArr = np.zeros((numbofSamples, TemporalLength))
    spatialArr = np.zeros((numbofSamples, 256))
    minrand = max(int(boxSize/2+1),int(patchSize/2))
    totalFeatures= np.zeros((numbofSamples,TemporalLength+256))

    #gets array with amount of features
    for i in range(numbofSamples):
        randx = random.randrange(minrand, width - minrand)
        randy = random.randrange(minrand, height - minrand)
        randz = random.randrange(int(TemporalLength/2), numFrames - int(TemporalLength/2))
        isWater[0,i] = mask[randy,randx]
        temporalArr[i,:] = features.fourierTransform(preprocessedVid,randx,randy,randz,boxSize,TemporalLength)
        temporalFeat = temporalArr[i,:]

        spatialArr[i,:] = features.SpatialFeatures(preprocessedVid,randx,randy,randz,patchSize,numFramesAvg)
        spaceFeat = spatialArr[i,:]
        combinedFeature = np.concatenate((np.reshape(temporalFeat,(temporalFeat.size)),spaceFeat))
        totalFeatures[i,:] = combinedFeature
    #plotFuncs.PlotTemporalFeatures(temporalArr, isWater,numbofSamples)
    #features.plotSpatialFeatures(spatialArr,isWater,numbofSamples)

    print("finished computing unified featureSpace")
    isWater = isWater.astype(int)
    return totalFeatures, isWater


def moduleB(vidpath,maskpath, numFrames, dFactor, densityMode,boxSize,NumbofFrameSearch,numbofSamples,patchSize,numFramesAvg):
    preprocess = preprocessVideo(vidpath,numFrames,dFactor,densityMode,1)
    features, isWater = getFeaturesPoints(preprocess,maskpath,dFactor,boxSize,NumbofFrameSearch,numbofSamples,patchSize,numFramesAvg)
    return features, isWater

#for training the classifier
def loopsThroughAllVids(pathToVidsFolder,pathToMasksPondFolder,pathToOtherTextures,numFrames, dFactor, densityMode,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg):
    #sets up vars
    totalFeatureSet = None
    isWateragg = None
    halfamountofVidsinFolder = 25
    #goes through first 20 vids
    for folders in os.listdir(pathToVidsFolder):
        counter = 0
        for vids in os.listdir(pathToVidsFolder + "/" + folders):
            print("INFO - Video Water: ", vids)
            if counter > halfamountofVidsinFolder:
                break
            nameMask = pathToMasksPondFolder + folders + "/" + vids[:-3] + 'png'
            nameVid = pathToVidsFolder + folders + "/" + vids
            #obtains features for video then concats them to matrix
            feature,isWater = moduleB(nameVid,nameMask,numFrames, dFactor, densityMode,boxSize,TemporalLength,numbofSamples,patchSize,numFramesAvg)
            if totalFeatureSet is None:
                totalFeatureSet = feature
            else:
                totalFeatureSet = np.concatenate((totalFeatureSet,feature),axis=0)
            if isWateragg is None:
                isWateragg = isWater
            else:
                isWateragg = np.concatenate((isWateragg,isWater),axis=1)
            counter+= 1
            print(counter)
        print(folders)
    halfamountofVidsinFolder = 20
    for folders in os.listdir(pathToOtherTextures):
        counter = 0
        for vids in os.listdir(pathToOtherTextures + folders):
            print("INFO - Video Non Water: ", vids)
            if counter > halfamountofVidsinFolder:
                break
            nameVid = pathToOtherTextures + folders + "/" + vids
            # obtains features for video then concats them to matrix
            feature, isWater = moduleB(nameVid, 0, numFrames, dFactor, densityMode, boxSize, TemporalLength,
                                       numbofSamples, patchSize, numFramesAvg)
            if totalFeatureSet is None:
                totalFeatureSet = feature
            else:
                totalFeatureSet = np.concatenate((totalFeatureSet, feature), axis=0)
            if isWateragg is None:
                isWateragg = isWater
            else:
                isWateragg = np.concatenate((isWateragg, isWater), axis=1)
            counter += 1
            print(counter)

    #turns water into correct orientation
    isWateragg = isWateragg * 1
    isWateragg = np.transpose(isWateragg)
    return totalFeatureSet, isWateragg
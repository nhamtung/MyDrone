import cv2
import numpy as np
import containerFunctions as ct
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

def main():
  VIDEO_WATER = "/content/drive/My Drive/water-detection/detectPuddle/dataset/videos/water/"
  VIDEO_NONWATER = "/content/drive/My Drive/water-detection/detectPuddle/dataset/videos/non_water/"
  MASK_WATER = "/content/drive/My Drive/water-detection/detectPuddle/dataset/masks/"

  DIR_MODEL = "/content/drive/My Drive/water-detection/detectPuddle/models/"
  MODEL_NAME = "tree50020410070001050.pkl"
  
  FRAME_EACH_BLOCK = 500
  DFACTOR = 2
  DENSITY_MODE = 0
  BOX_SIZE = 4
  TEMPORAL_LENGTH = 100
  NUMBER_OF_SAMPLES = 7000
  PATCH_SIZE = 10
  NUM_FRAME_AVG = 50
  
  # samples, y_labels = ct.loopsThroughAllVids(VIDEO_WATER, MASK_WATER, VIDEO_NONWATER, 500, 2, 0, 4, 200, 7000, 10, 74)
  samples, y_labels = ct.LoopsThroughAllVids(VIDEO_WATER, MASK_WATER, VIDEO_NONWATER, FRAME_EACH_BLOCK, DFACTOR, DENSITY_MODE, BOX_SIZE, TEMPORAL_LENGTH, NUMBER_OF_SAMPLES, PATCH_SIZE, NUM_FRAME_AVG)

  # np.save("saved_samples_deriv200.npy", samples)
  # np.save("saved_ylabels_deriv200.npy", y_labels)
  # samples = np.load("saved_samples_deriv200.npy")
  # y_labels = np.load("saved_ylabels_deriv200.npy")

  samples = samples.astype(np.float32)
  y_labels = y_labels.astype(int)
  y_labels = y_labels[:,0]
  print("samples.shape: ", samples.shape)
  print("y_labels.shape: ", y_labels.shape)
  
  model = RandomForestClassifier(n_estimators=40, n_jobs=-1, class_weight="balanced", max_features="log2")
  model.fit(samples, y_labels)
  joblib.dump(model, DIR_MODEL + MODEL_NAME)
  print("Saved model: ", DIR_MODEL)

if __name__ == '__main__':
   main()



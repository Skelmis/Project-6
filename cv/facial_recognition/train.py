from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join(
    [args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"]
)
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

from imutils import paths
import imutils
import pickle
from tools.FeatureExtractor import FeatureExtractor
from tools.Search import Search
import argparse
import cv2
import numpy as np

# Init feature extractor
fe = FeatureExtractor()

# Load image paths and features
imagePaths = sorted(list(paths.list_images("images")))
features = pickle.load(open("features.pickle", "rb"))

# Build arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to query image")
args = vars(ap.parse_args())

# Load query image, extract features and search closest vectors
img = cv2.imread(args["image"])
query = fe.extract(img)

res_images = Search.query(query, features, imagePaths)

# Loop over the closest images and show them
for image in res_images:
    image = cv2.imread(image)
    image = imutils.resize(image, width=400)
    cv2.imshow("result", image)
    cv2.waitKey(0)
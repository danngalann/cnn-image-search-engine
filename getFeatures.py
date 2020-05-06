import cv2
import pickle
from tools.FeatureExtractor import FeatureExtractor
from imutils import paths

fe = FeatureExtractor()
imagePaths = sorted(list(paths.list_images("images")))
features = []

# Extract the features for all the images
print("Extracting features...")
for i in range(len(imagePaths)):
    image = cv2.imread(imagePaths[i])
    ft = fe.extract(image)
    features.append(ft)
    print(f"Completed {i+1}/{len(imagePaths)}")

# And dump to file
pickle.dump(features, open("features.pickle", "wb"))

print("Done.")
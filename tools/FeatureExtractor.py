from keras.preprocessing import image as kimage
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        self.model = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", output_shape=[1280],
                        trainable=False)
        ])
        self.model.build([None, 224, 224, 3])

    def extract(self, img):
        img = cv2.resize(img, (224,224))
        img = kimage.img_to_array(img)
        x = np.expand_dims(img, axis=0) # Add a dimension

        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)  # Normalize
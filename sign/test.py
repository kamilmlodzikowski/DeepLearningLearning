import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import numpy
from tensorflow.keras.utils import to_categorical
import pickle
import time

liczba_klas=10

CATEGORIES = ["0", "1", "2","3","4", "5", "6", "7", "8", "9"]

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)
y_test= to_categorical(y_test, num_classes=liczba_klas)

print(y_test)

model = tf.keras.models.load_model("sign-CNN.model")
X_test = X_test/255.0

score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

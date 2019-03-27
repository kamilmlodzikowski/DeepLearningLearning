import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import numpy
from tensorflow.keras.utils import to_categorical
import pickle
import time

liczba_klas=6

CATEGORIES = ["d4", "d6", "d8","d10","d12", "d20"]

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)
y_test= to_categorical(y_test, num_classes=liczba_klas)

model = tf.keras.models.load_model("dice6-CNN.model")
X_test = X_test/255.0

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

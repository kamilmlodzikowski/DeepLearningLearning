import cv2
import tensorflow as tf
import os
from tqdm import tqdm
import numpy

DATADIR = "Valid"

CATEGORIES = ["0", "1", "2","3","4", "5", "6", "7", "8", "9"]

dobre=0
calosc=0

def prepare(filepath):
    IMG_SIZE = 100 
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array/255.0
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

model = tf.keras.models.load_model("sign-CNN.model")

for category in CATEGORIES: 

    path = os.path.join(DATADIR,category)  
    class_num = CATEGORIES.index(category)  

    for img in tqdm(os.listdir(path)):
        
        pathimg= os.path.join(path,img)
        prediction = model.predict([prepare(pathimg)])
        calosc=calosc+1
        przewi=prediction.tolist()[0]
        numpy.around(przewi,0)
       # try:
       #     print(przewi.index(1),CATEGORIES.index(category))
       # except ValueError:
       #     pass
        try:
            if przewi.index(1)==CATEGORIES.index(category):
                dobre=dobre+1
        except ValueError:
            calosc-=1
            pass

print("Dobre ", dobre, " na ", calosc)
procent= (float(dobre)/float(calosc))*100.0
print(numpy.round(procent,1), "%")

prediction2 = model.predict([prepare(pathimg)])
print(prediction2)  # will be a list in a list.
przewi=prediction.tolist()[0]
numpy.around(przewi,0)
try:
    print(CATEGORIES[przewi.index(1)])
except ValueError:
    pass

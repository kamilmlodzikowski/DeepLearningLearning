from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from keras.optimizers import SGD
from numpy import where
from tensorflow.keras import optimizers
import pickle
import time

liczba_klas=6


y= to_categorical(y, num_classes=liczba_klas) #num_classes odpowiada za liczbe klas
y_test= to_categorical(y_test, num_classes=liczba_klas)

X = X/255.0
X_test = X_test/255.0



model = Sequential()

# Step 1 - Convolutio Layer 
model.add(Conv2D(32, 3,  3, input_shape = (100, 100, 3), activation = 'relu'))

#step 2 - Pooling
model.add(MaxPooling2D(pool_size =(2,2)))

# Adding second convolution layer
model.add(Conv2D(32, 3,  3, activation = 'relu'))
model.add(MaxPooling2D(pool_size =(2,2)))




#Step 3 - Flattening
model.add(Flatten())

#Step 4 - Full Connection
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))	
model.add(Dense(liczba_klas, activation = 'softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(
              optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'train',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'valid',
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical')


modell = model.fit_generator(
        training_set,
        steps_per_epoch=800,
        epochs=8,
        validation_data = test_set,
        validation_steps = 6500
      )

model.save('dice6-CNN.model')

score = model.evaluate_generator(test_set)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


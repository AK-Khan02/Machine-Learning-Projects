import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
import cv2

train_data = pd.read_csv('sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist_test.csv')

train_data.info()

test_data.info()

train_label=train_data['label']
train_label.head()
trainset=train_data.drop(['label'],axis=1)
trainset.head()

X_train = trainset.values
X_train = trainset.values.reshape(-1,28,28,1)
print(X_train.shape)

test_label=test_data['label']
X_test=test_data.drop(['label'],axis=1)
print(X_test.shape)
X_test.head()

from sklearn.preprocessing import LabelBinarizer
lb=LabelBinarizer()
y_train=lb.fit_transform(train_label)
y_test=lb.fit_transform(test_label)
X_test=X_test.values.reshape(-1,28,28,1)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                  rotation_range = 0,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  shear_range=0,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  fill_mode='nearest')

X_test=X_test/255

fig,axe=plt.subplots(2,2)
fig.suptitle('Preview of dataset')
axe[0,0].imshow(X_train[0].reshape(28,28),cmap='gray')
axe[0,0].set_title('label: 3  letter: C')
axe[0,1].imshow(X_train[1].reshape(28,28),cmap='gray')
axe[0,1].set_title('label: 6  letter: F')
axe[1,0].imshow(X_train[2].reshape(28,28),cmap='gray')
axe[1,0].set_title('label: 2  letter: B')
axe[1,1].imshow(X_train[4].reshape(28,28),cmap='gray')
axe[1,1].set_title('label: 13  letter: M')

sns.countplot(train_label)
plt.title("Frequency of each label")

model=Sequential()
model.add(Conv2D(128,kernel_size=(5,5),
                 strides=1,padding='same',activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(3,3),strides=2,padding='same'))
model.add(Conv2D(64,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))
model.add(Conv2D(32,kernel_size=(2,2),
                strides=1,activation='relu',padding='same'))
model.add(MaxPool2D((2,2),2,padding='same'))

model.add(Flatten())

model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=24,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_datagen.flow(X_train,y_train,batch_size=200),
         epochs = 35,
          validation_data=(X_test,y_test),
          shuffle=1
         )

(ls,acc)=model.evaluate(x=X_test,y=y_test)

print('MODEL ACCURACY = {}%'.format(acc*100))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

# Load dataset
training_dataset = pd.read_csv('sign_mnist_train.csv')
testing_dataset = pd.read_csv('sign_mnist_test.csv')

# Dataset Information
training_dataset.info()
testing_dataset.info()

# Preparing training data
training_labels = training_dataset['label']
training_features = training_dataset.drop(['label'], axis=1)
reshaped_training_features = training_features.values.reshape(-1, 28, 28, 1)

# Preparing testing data
testing_labels = testing_dataset['label']
testing_features = testing_dataset.drop(['label'], axis=1)
reshaped_testing_features = testing_features.values.reshape(-1, 28, 28, 1)

# Label Binarization
label_converter = LabelBinarizer()
binarized_training_labels = label_converter.fit_transform(training_labels)
binarized_testing_labels = label_converter.transform(testing_labels)

# Normalize testing data
normalized_testing_features = reshaped_testing_features / 255

# Data Augmentation
augmentation_generator = ImageDataGenerator(rescale=1./255,
                                            rotation_range=0,
                                            height_shift_range=0.2,
                                            width_shift_range=0.2,
                                            shear_range=0,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            fill_mode='nearest')

# Preview dataset
figure, axes = plt.subplots(2, 2)
figure.suptitle('Preview of dataset')
axes[0, 0].imshow(reshaped_training_features[0].reshape(28, 28), cmap='gray')
axes[0, 0].set_title('label: 3  letter: C')
axes[0, 1].imshow(reshaped_training_features[1].reshape(28, 28), cmap='gray')
axes[0, 1].set_title('label: 6  letter: F')
axes[1, 0].imshow(reshaped_training_features[2].reshape(28, 28), cmap='gray')
axes[1, 0].set_title('label: 2  letter: B')
axes[1, 1].imshow(reshaped_training_features[4].reshape(28, 28), cmap='gray')
axes[1, 1].set_title('label: 13  letter: M')

# Label frequency plot
sns.countplot(training_labels)
plt.title("Frequency of each label")

# Building the model

sign_language_model = Sequential()
sign_language_model.add(Conv2D(128, kernel_size=(5, 5), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
sign_language_model.add(MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))
sign_language_model.add(Conv2D(64, kernel_size=(2, 2), strides=1, activation='relu', padding='same'))
sign_language_model.add(MaxPool2D((2, 2), 2, padding='same'))
sign_language_model.add(Conv2D(32, kernel_size=(2, 2), strides=1, activation='relu', padding='same'))
sign_language_model.add(MaxPool2D((2, 2), 2, padding='same'))
sign_language_model.add(Flatten())
sign_language_model.add(Dense(units=512, activation='relu'))
sign_language_model.add(Dropout(rate=0.25))
sign_language_model.add(Dense(units=24, activation='softmax'))
sign_language_model.summary()

# Compiling the model
sign_language_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
sign_language_model.fit(augmentation_generator.flow(reshaped_training_features, binarized_training_labels, batch_size=200),
                        epochs=35,
                        validation_data=(normalized_testing_features, binarized_testing_labels),
                        shuffle=1)

# Evaluate the model
loss_score, accuracy_score = sign_language_model.evaluate(x=normalized_testing_features, y=binarized_testing_labels)
print(f'MODEL ACCURACY = {accuracy_score * 100}%')

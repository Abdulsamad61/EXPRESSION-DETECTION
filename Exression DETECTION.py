# Make a expression detection which shows numbers when human is angry number should be in negative and increases according to anger	when he is happy mood then number should be positive input video stream of human and output should be labeled number according to their mood (+ve number = good mood -ve number =bad)	
#COURSE: MACHINE LEARNING
#BATCH: 2
# GROUP MEMBERS 
# ABDUL SAMAD 
# SHEHERYER ZIA SIDDIQUE
# MUHAMMAD TALHA ASHRAF

import sys, os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# IMPORT CSV FILE FOR TRAINING
df=pd.read_csv('fer2013.csv')

print(df.info())
print(df["Usage"].value_counts())

print(df.head())
X_train,train_y,X_test,test_y=[],[],[],[]
# SPLITING PROCESS
for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    try:
        if 'Training' in row['Usage']:
           X_train.append(np.array(val,'float32'))
           train_y.append(row['emotion'])
        elif 'PublicTest' in row['Usage']:
           X_test.append(np.array(val,'float32'))
           test_y.append(row['emotion'])
    except:
        print(f"error occured at index :{index} and row:{row}")
num_features = 64
num_labels = 7
batch_size = 64
epochs = 350
width, height = 48, 48

# TRAINING AND TESTING PROCESS BETWEEN X & Y
X_train = np.array(X_train,'float32')
train_y = np.array(train_y,'float32')
X_test = np.array(X_test,'float32')
test_y = np.array(test_y,'float32')

train_y=np_utils.to_categorical(train_y, num_classes=num_labels)
test_y=np_utils.to_categorical(test_y, num_classes=num_labels)

#cannot produce
#NORMALIZING DATA BETWEEN 0 AND 1
X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)

X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)

X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# print(f"shape:{X_train.shape}")

#DESIGNING THE CNN

#1st CONVOLUTION LAYER
model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1:])))
model.add(Conv2D(64,kernel_size= (3, 3), activation='relu'))

# MODEL.ADD(BATCH NORMALIZATION())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#2nd CONVOLUTION LAYER

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))

#  MODEL.ADD(BATCH NORMALIZATION())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
model.add(Dropout(0.5))

#3rd CONVOLUTION LAYER
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))

# MODEL.ADD(BATCH NORMALIZATION())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Flatten())

# FULLY CONNECTED NEURAL NETWORK

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

# MODEL.SUMMARY()

#COMPILING THE MODEL

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(),
              metrics=['accuracy'])

#TRAINING THE MODEL 

model.fit(X_train, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, test_y),
          shuffle=True)
#SHOWING TRAINNING ACCURACY
Train_acc = model.evaluate(X_train,train_y,verbose=0)
print('Train accuracy:',100*Train_acc[1])
#SHOWING TESTING ACCURACY
test_acc = model.evaluate(X_test,test_y,verbose=0)
print('Test accuracy:',100*Train_acc[1])

#SAVING THE MODEL TO USE IT LATER ON

fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
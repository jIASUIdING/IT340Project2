#!/usr/bin/env python
# coding: utf-8

# ### Import and prepare data

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# Set the number of epochs here (not specified in assignment
# so adjust as needed)
epochs = 5

# the data, split between train and test sets
(x_training_data, y_training_data), (x_testing_data, y_testing_data) = mnist.load_data()

# 784 is the number of pixels in a 28x28 image
x_training_data = x_training_data.reshape(60000, 784)
x_testing_data = x_testing_data.reshape(10000, 784)

# Change the training/testing data to be between 0 and 1
x_training_data = x_training_data.astype('float32')
x_testing_data = x_testing_data.astype('float32')
x_training_data /= 255
x_testing_data /= 255

# Convert to categorical
y_training_data = keras.utils.to_categorical(y_training_data, 10)
y_testing_data = keras.utils.to_categorical(y_testing_data, 10)


# ### Model 1: A 3-layered MLP with all hidden layers have a size of 256.

# Add input layer
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))

# Add hidden layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

# Softmax the result (output layer)
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Fit the model with the training data
model.fit(x_training_data, y_training_data,
                    batch_size=128,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testing_data, y_testing_data))


# ### Model 2: A 2-layered MLP with all hidden layers have a size of 512.

# Add input layer
model2 = Sequential()
model2.add(Dense(32, activation='relu', input_shape=(784,)))
model2.add(Dropout(0.2))

# Add hidden layers
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.2))

# Softmax the result (output layer)
model2.add(Dense(10, activation='softmax'))

# Compile the model2
model2.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Fit the model2 with the training data
model2.fit(x_training_data, y_training_data,
                    batch_size=128,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testing_data, y_testing_data))


# ### Model 3: A 3-layered MLP with the hidden layers’ sizes being 128, 256, and 512.

# Add input layer
model3 = Sequential()
model3.add(Dense(32, activation='relu', input_shape=(784,)))
model3.add(Dropout(0.2))

# Add hidden layers
model3.add(Dense(128, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.2))

# Softmax the result (output layer)
model3.add(Dense(10, activation='softmax'))

# Compile the model3
model3.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Fit the model3 with the training data
model3.fit(x_training_data, y_training_data,
                    batch_size=128,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testing_data, y_testing_data))


# ### Model 4: A 3-layered MLP with the hidden layers’ sizes being 512, 256, and 128.

# Add input layer
model4 = Sequential()
model4.add(Dense(32, activation='relu', input_shape=(784,)))
model4.add(Dropout(0.2))

# Add hidden layers
model4.add(Dense(512, activation='relu'))
model4.add(Dropout(0.2))
model4.add(Dense(256, activation='relu'))
model4.add(Dropout(0.2))
model4.add(Dense(128, activation='relu'))
model4.add(Dropout(0.2))

# Softmax the result (output layer)
model4.add(Dense(10, activation='softmax'))

# Compile the model4
model4.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Fit the model4 with the training data
model4.fit(x_training_data, y_training_data,
                    batch_size=128,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testing_data, y_testing_data))


# ### Evaluate

# Evaluate the accuracy using the testing data
score = model.evaluate(x_testing_data, y_testing_data, verbose=0)
print('Accuracy of 2-layered MLP with all hidden layers with size of 512: ', score[1])

score = model2.evaluate(x_testing_data, y_testing_data, verbose=0)
print('Accuracy of 2-layered MLP with all hidden layers with size of 512: ', score[1])

score = model3.evaluate(x_testing_data, y_testing_data, verbose=0)
print('Accuracy of 3-layered MLP with hidden layers sized 128, 256, 512: ' , score[1])

score = model4.evaluate(x_testing_data, y_testing_data, verbose=0)
print('Accuracy of 3-layered MLP with hidden layers sized 512, 256, 128: ' , score[1])


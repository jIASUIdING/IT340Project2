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

# Add input layer
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))

# Add hidden layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
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

# Evaluate the accuracy using the testing data
score = model.evaluate(x_testing_data, y_testing_data, verbose=0)
print('Accuracy of 3-layered MLP with hidden layers sized 512, 256, 128: ' + score[1])
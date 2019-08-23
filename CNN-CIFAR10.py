from __future__ import print_function

import keras
from keras.datasets import mnist, cifar10
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

import os
import numpy as np
import cv2 as cv

import sys
sys.path.append('D:/Projects/DeepLearning/mascml/')
import mascml
from keras.utils import plot_model

def CNN_Cifar10(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


batch_size = 32
num_classes = 10
epochs = 40
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'save_models')
model_name = 'keras_cifar10_trained_model_earlystop3.h5'
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

trainFlag = 1
print(x_train.shape)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if trainFlag:

    # creatModel
    model = CNN_Cifar10(x_train.shape[1:])
    #plot_model(model, to_file='model.png')

    # initial RMSprop
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # compile
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # callback
    from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping

    callbacks = []
    modelChecks = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    earlyStop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    callbacks = [modelChecks, earlyStop]

    # fit
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                     callbacks=callbacks, shuffle=True)  #
    mascml.utils.plot_history(hist)
    model.save(model_path)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
else:
    idx = 0
    model = load_model(model_path)

    prob = model.predict(x_test)
    yhat = (np.argmax(keras.utils.to_categorical(np.round(prob), num_classes), axis=1))
    y_test = (np.argmax(keras.utils.to_categorical(y_test, num_classes), axis=1))
    yhat = np.sum(yhat, axis=1)
    y_test = np.sum(y_test, axis=1)

    IMG = np.reshape(x_test[idx, :, :, :], [28, 28])
    cv.imshow(" pre:" + str(yhat[idx]) + " prob:" + str(prob[idx, yhat[idx]]), IMG)
    cv.waitKey(0)

    print(IMG.shape)

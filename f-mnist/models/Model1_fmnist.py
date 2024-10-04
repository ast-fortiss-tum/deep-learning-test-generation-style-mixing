'''
https://gist.github.com/kashif/76792939dd6f473b7404474989cb62a8
'''

from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Dropout, Activation, Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import os
import time
import numpy as np

def Model1_fmnist(input_tensor=None, train=False):
    # input image dimensions
    img_rows, img_cols, img_chn = 28, 28, 1
    input_shape = (img_rows, img_cols, img_chn)
    if train:
        # start_time = time.clock()
        batch_size = 128
        num_classes = 10
        epochs = 30

        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        # Normalize data.
        x_train = x_train.astype('float32').reshape((-1, 28, 28, 1))
        x_train = x_train/ 255
        x_test = x_test.astype('float32').reshape((-1, 28, 28, 1))
        x_test = x_test/ 255

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        input_tensor = Input(shape=input_shape)
    # Model definition
    x = Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1))(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax')(x)

    model = Model(input_tensor, x)
    if train:
        # compiling
        model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                validation_data=(x_test, y_test), verbose=2, shuffle=True)
        # save model
        model.save_weights('./Model1_fmnist.h5')
        model.save('./Model1_fmnist_full.h5')
        score = model.evaluate(x_test, y_test, verbose=0)
        print('\n')
        print('Overall Test score:', score[0])
        print('Overall Test accuracy:', score[1])
        # Printing out training execution time
        # print("--- %s seconds ---" % (time.clock() - start_time))
    else:
        model.load_weights('/Users/giorgi/Desktop/kloni/Test_models/f-mnist/models/Model1_fmnist.h5')
        print('FMNIST Model1 loaded')
    return model

if __name__ == '__main__':
    Model1_fmnist(train=False)

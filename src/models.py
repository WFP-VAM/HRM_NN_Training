from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten, BatchNormalization
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.applications.vgg16 import VGG16
import tensorflow as tf


def google_net(size=256, kernel=3):
    model = Sequential()
    model.add(Conv2D(32, (kernel, kernel), 
                     activation='relu', 
                     input_shape=(size, size, 3),
                     strides=2,
                     kernel_regularizer=regularizers.l2(0.01),
                     name='cv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (kernel, kernel), 
                     activation='relu',
                     strides=2,
                     kernel_regularizer=regularizers.l2(0.01),
                     name='cv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (kernel, kernel), activation='relu', strides=2, name='cv3.3'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01),  name='features'))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax', name='denseout'))

    print(model.summary())

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-4, decay=0.1e-6),
        metrics=['accuracy'])

    return model


def sentinel_net(size=400, kernel=3):
    model = Sequential()
    model.add(Conv2D(32, (kernel, kernel), activation='relu', input_shape=(size, size, 3), name='cv1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (kernel, kernel), activation='relu', name='cv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (kernel, kernel), activation='relu', name='cv3'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, name='features'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax', name='denseout'))

    print(model.summary())

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-3),#, decay=0.1e-5),
        metrics=['accuracy'])

    return model


def google_vgg16_finetune(classes=3, size=256):

    input_layer = Input(shape=(size, size, 3), name='image_input')
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)
    model = Sequential()
    for layer in base_model.layers:
        model.add(layer)

    model.add(Flatten(name='avgpool'))
    model.add(Dense(256, activation='relu', name='features', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(classes, activation='softmax', name='out'))

    for layer in model.layers:
        if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3', 'features', 'out']:
            layer.trainable = True
        else:
            layer.trainable = False

    print(model.summary())

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
        metrics=['accuracy'])

    return model
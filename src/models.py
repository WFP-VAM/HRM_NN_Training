from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten, BatchNormalization
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.mobilenet import MobileNet
import tensorflow as tf


def google_net(size=256, kernel=3):
    model = Sequential()
    model.add(Conv2D(32, (kernel, kernel), 
                     activation='relu', 
                     input_shape=(size, size, 3),
                     strides=1,
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer='he_uniform',
                     name='cv1'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (kernel, kernel),
                     activation='relu',
                     input_shape=(size, size, 3),
                     strides=1,
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer='he_uniform',
                     name='cv2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (kernel, kernel),
                     activation='relu',
                     strides=1,
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer='he_uniform',
                     name='cv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (kernel, kernel),
                     activation='relu',
                     strides=1,
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer='he_uniform',
                     name='cv4'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256,
                    kernel_regularizer=regularizers.l2(0.01),
                    kernel_initializer='he_uniform',
                    name='features'))
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
    model.add(Conv2D(32, (kernel, kernel),
                     activation='relu',
                     input_shape=(size, size, 3),
                     strides=2,
                     kernel_regularizer=regularizers.l2(0.01),
                     name='cv1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (kernel, kernel),
                     activation='relu',
                     strides=2,
                     kernel_regularizer=regularizers.l2(0.01),
                     name='cv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (kernel, kernel),
                     activation='relu',
                     strides=2,
                     kernel_regularizer=regularizers.l2(0.01),
                     name='cv3'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=regularizers.l2(0.01), name='features'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax', name='denseout'))

    print(model.summary())

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-4, decay=0.1e-6),
        metrics=['accuracy'])

    return model


def vgg16_finetune(classes=3, size=256):

    input_layer = Input(shape=(size, size, 3), name='image_input')
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer)

    x = Conv2D(name='squeeze', filters=256, kernel_size=(1, 1))(base_model.output)  # squeeze channels
    x = Flatten(name='avgpool')(base_model.output)
    x = Dense(256, name='features', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(classes, activation='softmax', name='out')(x)

    model = Model(inputs=base_model.input, outputs=x)
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


def google_mobnet_finetune(classes=3, size=256):
    input_layer = Input(shape=(size, size, 3), name='image_input')
    base_model = MobileNet(weights='imagenet', include_top=False, input_tensor=input_layer)

    x = Conv2D(name='squeeze', filters=256, kernel_size=(1,1))(base_model.output)  # squeeze channels
    x = Dropout(0.5)(x)
    x = Flatten(name='avgpool')(x)
    x = Dense(256, name='features', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(classes, activation='softmax', name='out')(x)
    model = Model(inputs=base_model.input, outputs=x)

    for layer in model.layers:
        if layer.name in ['conv_pw_13', 'conv_pw_13_bn', 'squeeze', 'features', 'out']:
            layer.trainable = True
        else:
            layer.trainable = False

    print(model.summary())

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
        metrics=['accuracy'])

    return model


def google_simple(size=256):
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     kernel_initializer='he_uniform',
                     input_shape=(size, size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # extra
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(3, activation='softmax', name='denseout'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
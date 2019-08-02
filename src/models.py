from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, \
    Activation, Dense, Flatten, BatchNormalization, Dropout
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.optimizers import RMSprop, SGD
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.applications.vgg16 import VGG16
import tensorflow as tf


def njean_google():
    """
    replicating the model used in NJean's paper.
    https://github.com/nealjean/predicting-poverty/blob/master/model/predicting_poverty_deploy.prototxt

    Its a VGG-F with 1d conv layers instead of the dense at the end. Original paper:
    http://www.bmva.org/bmvc/2014/files/paper054.pdf

    NJ uses 400x400, zoom 16, batches of 32, 3 channels.

    Weights are available in caffe, we converted to h5.
    """
    model = Sequential()

    # first conv block
    model.add(Conv2D(
        filters=64,
        input_shape=(400, 400, 3),
        kernel_size=(11,11),
        strides=(4,4),
        kernel_initializer='glorot_uniform',
        name="conv1",
        activation='relu'
    ))

    # NJean uses LRN but lets try with batch norm on the channels
    model.add(BatchNormalization(
        axis=3, name='norm1'
    ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='pool1'))

    # second conv block
    model.add(Conv2D(
        filters=256,
        padding='same',
        kernel_size=(5, 5),
        kernel_initializer='glorot_uniform',
        name="conv2",
        activation='relu'
    ))

    model.add(BatchNormalization(
        axis=3, name='norm2'
    ))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='pool2'))

    # third conv block, no pool
    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='glorot_uniform',
        name="conv3",
        activation='relu'
    ))

    # not in NJ
    model.add(Dropout(0.5))

    # fourth conv block, no pooling
    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='glorot_uniform',
        name="conv4",
        activation='relu'
    ))

    # not in NJ
    model.add(Dropout(0.5))

    # fifth conv block
    model.add(Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer='glorot_uniform',
        name="conv5",
        activation='relu'
    ))

    # not in NJ
    model.add(Dropout(0.5))

    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, name='pool5'))

    # sixth conv block, replacing the fully dense
    model.add(Conv2D(
        filters=4096,
        kernel_size=(6, 6),
        strides=(6,6),
        padding='valid',
        kernel_initializer='glorot_uniform',
        name="conv6",
        activation='relu'
    ))

    # not in NJ
    model.add(Dropout(0.5))

    # seventh conv block, replacing the fully dense
    model.add(Conv2D(
        filters=4096,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer='glorot_uniform',
        name="conv7",  # features of NJean
        activation='relu'
    ))

    # not in NJ
    model.add(Dropout(0.5))

    # eights conv block, replacing the dense output. no activation
    model.add(Conv2D(
        filters=3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        kernel_initializer='glorot_uniform',
        name="conv8"
    ))
    # how can I pool on 1 values dim?
    #model.add(AveragePooling2D(pool_size=(2, 2), strides=1, name='pool6'))

    model.add(Flatten())
    model.add(Activation('softmax', name='prob'))

    # for layer in model.layers:
    #     if layer.name in ['conv5', 'conv6', 'conv7', 'conv8']:
    #         layer.trainable = True
    #         print('trainable layer: ', layer.name)
    #     else:
    #         layer.trainable = False

    print(model.summary())

    sgd = SGD(
        lr=1e-6,
        decay=1e-9,
        momentum=0.9,
        nesterov=True)
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

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
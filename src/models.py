from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten, BatchNormalization
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from src.loss import basic_loss

def netowrk(size):
    inputs = Input((size, size, 3))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    conv1 = BatchNormalization(axis=3)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    pool2 = Dropout(0.25)(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)

    dense1 = Dense(256)(pool3)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(0.25)(dense1)

    out = Flatten(name='flatten')(dense1)
    out = Dense(3)(out)
    out = Activation('softmax')(out)

    model = Model(inputs=[inputs], outputs=[out])

    # compile and train ----------------------------------------------
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=0.000001),
                  metrics=['accuracy'])

    return model


def baseline_net(size=400, kernel=3):
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

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-3),#, decay=0.1e-5),
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

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-3),#, decay=0.1e-5),
        metrics=['accuracy'])

    return model


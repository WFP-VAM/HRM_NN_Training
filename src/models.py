from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, Dense, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import RMSprop
from src.loss import basic_loss


def baseline_net(size=400, kernel=3):
    model = Sequential()
    model.add(Conv2D(32, (kernel, kernel), activation='relu', input_shape=(size, size, 3), name='cv1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (kernel, kernel), activation='relu', name='cv2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (kernel, kernel), activation='relu', name='cv3'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, name='features'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax', name='denseout'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(lr=1e-3),#, decay=0.1e-5),
        metrics=['accuracy'])

    return model

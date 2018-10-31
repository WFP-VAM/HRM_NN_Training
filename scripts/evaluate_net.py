""" Calculating the accuracy of the model """
from PIL import Image
import os
import tensorflow as tf
import numpy as np

# get the list of files
data_list = []
for root, dirs, files in os.walk('data/Google/test/'):
    for name in files:
        if name.endswith((".png")):
            data_list.append(root+'/'+name)

# load an image
def get_image(image_path):
    image = Image.open(image_path, 'r')
    image = image.convert('RGB').resize((400,400))
    image = np.array(image) / 255.
    return image

# load images
def load_images(files, directory):
    images = []
    for f in files:
        im = get_image(directory + f)
        images.append(im)
    return images

# run
X = np.array(load_images(data_list, ''))

# load model and predict
model = tf.keras.models.load_model('results/nightGoo.h5')
y_pred = model.predict(X)
y_true = [1]*300 + [2]*300 + [3]*300

# classification report
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
le = LabelBinarizer()
y = le.fit_transform(np.array(y_true))
print(classification_report(y, np.round(y_pred)))
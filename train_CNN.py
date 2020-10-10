"""
Author: Suhas Cholleti (sc3614@g.rit.edu)
Author: Byreddy Vishnu (vb38914@g.rit.edu)
Author: Pranjal Pandey (pp9034@g.rit.edu)

This program takes a dataset and stores the output in the given output path.
It divides the given dataset into 2 parts for prediction and learning. It gives the
detailed analysis of the prediction.
"""

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os


def get_data_of_Images(dataset):
    """
    Iterates over a list of images and returns the features and labels of all the images.
    It resizes the image to 32x32 pixels and flattens the image into 32x32x3.
    We also scale the color range from [0, 255] to [0, 1]
    """
    data, labels = [], []
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    # shuffling the images randomly
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)
    return data, labels


# Parsing the inputs
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="output path for trained model")
ap.add_argument("-l", "--label-bin", required=True, help="output path for label binarizer")
args = vars(ap.parse_args())

# initialize the data and labels
print("Loading images...")
data, labels = get_data_of_Images(args["dataset"])

# Split dataset into training and testing data set. 25% for testing, 75% for training.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 30

# compile the model using SGD as our optimizer and categorical cross-entropy loss
print("Training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the neural network
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# evaluating the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# save the model and label binarizer to future predictions
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

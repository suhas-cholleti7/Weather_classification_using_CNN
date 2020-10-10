"""
Author: Suhas Cholleti (sc3614@g.rit.edu)
Author: Byreddy Vishnu (vb38914@g.rit.edu)
Author: Pranjal Pandey (pp9034@g.rit.edu)

This program takes a dataset and a model name and runs the specific model on the dataset
It divides the given dataset into 2 parts for prediction and learning. It gives the
details analysis of the prediction.
"""

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os


models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
}

def extract_features(image):
    """
    Takes an PIL.Image object as an input and divides the image into three channels.
    It returns the mean and standard division of the each channel as the six
    features of teh image.
    """
    bands = image.getbands()
    if len(bands) == 1:
        return "Single"
    (R, G, B) = image.split()
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
                np.std(G), np.std(B)]
    return features


def get_features_of_Images(dataset):
    """
    Iterates over a list of images and returns the features and labels of all the images.
    """
    imagePaths = paths.list_images(dataset)
    data = []
    labels = []
    for imagePath in imagePaths:
        image = Image.open(imagePath)
        print(imagePath.split(os.path.sep)[-1])
        features = extract_features(image)
        if features == "Single":
            continue
        data.append(features)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    return data, labels


# Parsing the inputs
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="weather_classification",
                help="path to directory containing the '3scenes' dataset")
ap.add_argument("-m", "--model", type=str, default="naive_bayes",
                help="type of python machine learning model to use")
args = vars(ap.parse_args())


print("Extracting image features...")
data, labels = get_features_of_Images(args["dataset"])

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split dataset into training and testing data set. 25% for testing, 75% for training.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

# train the model
print("Using '{}' model".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)

# make predictions on our data and show a classification report
print("Evaluating...")
predictions = model.predict(testX)
print(classification_report(testY, predictions,
                            target_names=le.classes_))

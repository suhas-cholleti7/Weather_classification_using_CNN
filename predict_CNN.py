"""
Author: Suhas Cholleti (sc3614@g.rit.edu)
Author: Byreddy Vishnu (vb38914@g.rit.edu)
Author: Pranjal Pandey (pp9034@g.rit.edu)

This program takes an image and predicts the weather in the image.
"""

# import the necessary packages
from keras.models import load_model
import argparse
import pickle
import cv2

HEIGHT, WIDTH = 32, 32


def resize_reshape_image(image_path):
	"""
	load the input image and resize it to the target dimensions
	We also scale the color range from [0, 255] to [0, 1]
	"""
	image = cv2.imread(image_path)
	output = image.copy()
	image = cv2.resize(image, (WIDTH, HEIGHT))
	image = image.astype("float") / 255.0
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	return image, output


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image for classification")
ap.add_argument("-m", "--model", required=True,
	help="path to trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
args = vars(ap.parse_args())


image, output = resize_reshape_image(args["image"])

print("Loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())

# make a prediction on the image
preds = model.predict(image)

# find the class label index with the largest corresponding probability
i = preds.argmax(axis=1)[0]
label = lb.classes_[i]

# draw the class label and probability on the image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 0, 255), 2)

cv2.imshow("Image", output)
cv2.imwrite("results/" + args["image"], output)
cv2.waitKey(0)
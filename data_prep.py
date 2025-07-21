import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "asl_alphabet_train")

IMG_SIZE = 64

def load_data():
    images, labels = [], []
    classes = sorted(os.listdir(DATA_DIR))

    for label, sign in enumerate(classes):
        path = os.path.join(DATA_DIR, sign)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), classes

X, y, class_names = load_data()
X = X / 255.0

np.save("X.npy", X)
np.save("y.npy", y)
with open("class_names.txt", "w") as f:
    f.write(",".join(class_names))

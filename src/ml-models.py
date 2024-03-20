from google.colab.output import eval_js

import time
start_time = time.time()

!pip install tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.notebook import tqdm

import keras
from keras import backend as K
from tensorflow.keras.layers import *
from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape

!pip install scikeras
from scikeras.wrappers import KerasClassifier

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
import random
from PIL import Image
import gdown

import argparse
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D, Add, Concatenate
from keras.models import Model
import struct
from google.colab.patches import cv2_imshow
from copy import deepcopy
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from keras.applications.mobilenet import MobileNet

!pip install hypopt
from hypopt import GridSearch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

!pip install -U opencv-contrib-python
import cv2

!pip install tensorflowjs
import tensorflowjs as tfjs

from google.colab import files

import requests, io, zipfile

# Prepare data
os.makedirs('images_1', exist_ok=True)
os.makedirs('images_2', exist_ok=True)
os.makedirs('images_all', exist_ok=True)

metadata_path = 'metadata.csv'
image_path_1 = 'images_1.zip'
image_path_2 = 'images_2.zip'
images_rgb_path = 'hmnist_8_8_RGB.csv'

!wget -O metadata.csv 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/metadata.csv'
!wget -O images_1.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_1.zip'
!wget -O images_2.zip 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/images_2.zip'
!wget -O hmnist_8_8_RGB.csv 'https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20(Healthcare%20B)%20Skin%20Cancer%20Diagnosis/hmnist_8_8_RGB.csv'
!unzip -q -o images_1.zip -d images_1
!unzip -q -o images_2.zip -d images_2

!pip install patool
import patoolib

import os.path
from os import path

from distutils.dir_util import copy_tree

fromDirectory = 'images_1'
toDirectory = 'images_all'

copy_tree(fromDirectory, toDirectory)

fromDirectory = 'images_2'
toDirectory = 'images_all'

copy_tree(fromDirectory, toDirectory)

IMG_WIDTH = 100
IMG_HEIGHT = 75

X = []
X_gray = []

y = []

metadata = pd.read_csv(metadata_path)
metadata['category'] = metadata['dx'].replace({'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6,})

for i in tqdm(range(len(metadata))):
    image_meta = metadata.iloc[i]
    path = os.path.join(toDirectory, image_meta['image_id'] + '.jpg')
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))

    img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    X_gray.append(img_g)

    X.append(img)
    y.append(image_meta['category'])

X_gray = np.array(X_gray)
X = np.array(X)
y = np.array(y)

print(X_gray.shape)
print(X.shape)
print(y.shape)

cv2_imshow(X_gray[10014])

sample_cap = 142
option = 1

if option == 1:
    objects = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    class_totals = [0,0,0,0,0,0,0]
    iter_samples = [0,0,0,0,0,0,0]
    indicies = []

    for i in range(len(X)):
        class_totals[y[i]] += 1

    print("Initial Class Samples")
    print(class_totals)

    for i in range(len(X)):
        if iter_samples[y[i]] != sample_cap:
            indicies.append(i)
            iter_samples[y[i]] += 1

    X = X[indicies]
    X_gray = X_gray[indicies]

    y = y[indicies]

    class_totals = [0,0,0,0,0,0,0]

    for i in range(len(X)):
        class_totals[y[i]] += 1

    print("Modified Class Samples")
    print(class_totals)

X_gray_train, X_gray_test, y_train, y_test = train_test_split(X_gray, y, test_size=0.4, random_state=101)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

X_augmented = []
X_gray_augmented = []

y_augmented = []

for i in tqdm(range(len(X_train))):
    transform = random.randint(0,1)
    if transform == 0:
        flipped_image = None
        flipped_image_gray = None

        X_augmented.append(flipped_image)
        X_gray_augmented.append(flipped_image_gray)
        y_augmented.append(y_train[i])
    else:
        zoomed_image = None
        zoomed_image_gray = None

        X_augmented.append(zoomed_image)
        X_gray_augmented.append(zoomed_image_gray)
        y_augmented.append(y_train[i])

knn = KNeighborsClassifier(n_neighbors=5)

X_g_train_flat = X_gray_train.reshape(X_gray_train.shape[0],-1)
X_g_test_flat = X_gray_test.reshape(X_gray_test.shape[0],-1)
print (X_g_train_flat.shape)
print (X_g_test_flat.shape)

knn.fit(X_g_train_flat, y_train)

def model_stats(name, y_test, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test, y_pred)

    print(name)

    accuracy = accuracy_score(y_test,y_pred)
    print ("The accuracy of the model is " + str(round(accuracy,5)))

    roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')

    print ("The ROC AUC Score of the model is " + str(round(roc_score,5)))

    return cm

y_pred = knn.predict(X_g_test_flat)
y_pred_proba = knn.predict_proba(X_g_test_flat)

knn_cm = model_stats("K Nearest Neighbors",y_test,y_pred,y_pred_proba)

def plot_cm(name, cm):
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
    df_cm = df_cm.round(5)

    plt.figure(figsize = (12,8))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title(name + " Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

plot_cm("K Nearest Neighbors",knn_cm)

def CNNClassifier_Modified(epochs=20, batch_size=10, layers=5, dropout=0.5, activation='relu'):
    def set_params():
        i = 1
    def create_model():
        model = Sequential()

        model.add(Flatten())
        model.add(Dense(7))
        model.add(Activation('softmax'))

        opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[tf.keras.metrics.AUC()])
        return model
    return KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)

y_train_onehot = np.zeros((y_train.size, y_train.max().astype(int)+1))
y_train_onehot[np.arange(y_train.size),y_train.astype(int)] = 1

y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int)+1))
y_test_onehot[np.arange(y_test.size),y_test.astype(int)] = 1

cnn = CNNClassifier_Modified()

cnn.fit(X_train.astype(np.float32), y_train_onehot.astype(np.float32),
        validation_data=(X_test.astype(np.float32),y_test_onehot.astype(np.float32)),verbose=1)

tfjs.converters.save_keras_model(cnn.model, 'cnn_model')

def model_stats(name, y_test, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test, y_pred)

    print(name)

    accuracy = accuracy_score(y_test,y_pred)
    print ("The accuracy of the model is " + str(round(accuracy,5)))

    y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int)+1))
    y_test_onehot[np.arange(y_test.size),y_test.astype(int)] = 1

    roc_score = roc_auc_score(y_test_onehot, y_pred_proba)

    print ("The ROC AUC Score of the model is " + str(round(roc_score,5)))

    return cm

y_pred = cnn.predict(X_test)
y_pred_proba = cnn.predict_proba(X_test)
cnn_cm = model_stats("CNN",y_test,y_pred,y_pred_proba)

def plot_cm(name, cm):
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
    df_cm = df_cm.round(5)

    plt.figure(figsize = (12,8))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title(name + " Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

plot_cm("CNN", cnn_cm)

dtc = DecisionTreeClassifier(max_depth=5)

dtc = dtc.fit(X_g_train_flat, y_train)

def model_stats(name, y_test, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test, y_pred)

    print(name)

    accuracy = accuracy_score(y_test,y_pred)
    print ("The accuracy of the model is " + str(round(accuracy,5)))

    roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')

    print ("The ROC AUC Score of the model is " + str(round(roc_score,5)))

    return cm

y_pred = dtc.predict(X_g_test_flat)
y_pred_proba = dtc.predict_proba(X_g_test_flat)
dtc_cm = model_stats("DTC Accuracy:",y_test,y_pred,y_pred_proba)

def plot_cm(name, cm):
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
    df_cm = df_cm.round(5)

    plt.figure(figsize = (12,8))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title(name + " Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

plot_cm("DTC", dtc_cm)

rfc = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
rfc = rfc.fit(X_g_train_flat, y_train)

def model_stats(name, y_test, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test, y_pred)

    print(name)

    accuracy = accuracy_score(y_test,y_pred)
    print ("The accuracy of the model is " + str(round(accuracy,5)))

    roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')

    print ("The ROC AUC Score of the model is " + str(round(roc_score,5)))

    return cm

y_pred = rfc.predict(X_g_test_flat)
y_pred_proba = rfc.predict_proba(X_g_test_flat)
rfc_cm = model_stats("RFC Accuracy:",y_test,y_pred,y_pred_proba)

def plot_cm(name, cm):
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
    df_cm = df_cm.round(5)

    plt.figure(figsize = (12,8))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title(name + " Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

plot_cm("RFC", rfc_cm)
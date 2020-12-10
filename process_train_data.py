import cv2
import numpy as np
import os

from random import shuffle
from tqdm import tqdm

TRAIN_DIR = os.getcwd() + '/train'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

def label_img(img):
    word_label = img[0]
    if word_label == 'b': return [1,0,0,0]
    elif word_label == 'h': return [0,1,0,0]
    elif word_label == 'l': return [0,0,1,0]
    elif word_label == 'v': return [0,0,0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_UNCHANGED), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

create_train_data()

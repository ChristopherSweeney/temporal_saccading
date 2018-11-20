#-----------------------------------------------------------------------------------------------------------------------
# Project: resnet-finetune-demo
# Filename: build_features.py
# Date: 16.06.2017
# Author: Adam Brzeski - CTA.ai
#-----------------------------------------------------------------------------------------------------------------------

"""
Builds cached Resnet features for train, val and test subsets of Indoor-67 dataset. Cached features are saved to disk
as numpy arrays. Except features, labels infered from paths and paths itself are also saved.

In short, script will create 3 files in each subset directory (train, val, test):
- features-resnet152.npy
- labels-resnet152.npy
- paths-resnet152.json

"""

import glob
import json
import os
import numpy as np
import skimage.io
from keras.models import Model
import helper
from resnet import resnet152
import matplotlib.pyplot as plt


DATA_SUBSETS = [
    os.path.expanduser("~/ml/data/indoor/train"),
    os.path.expanduser("~/ml/data/indoor/val"),
    os.path.expanduser("~/ml/data/indoor/test"),
]
FEATURES_FILENAME = "features-resnet152.npy"
LABELS_FILENAME = "labels-resnet152.npy"
PATHS_FILENAME = "paths-resnet152.json"
WEIGHTS_RESNET = os.path.expanduser("~/ml/models/keras/resnet152/resnet152_weights_tf.h5")
NAMES_TO_IDS = json.load(open("names_to_ids.json"))


# Load Resnet 152 model and construct feature extraction submodel
resnet_model = resnet152.resnet152_model(WEIGHTS_RESNET)
feature_layer = 'avg_pool'
features_model = Model(inputs=resnet_model.input,
                       outputs=resnet_model.get_layer(feature_layer).output)

def saliency_queue(img,size,scale):
    output=[]
    h,w = np.shape(img)
    for i in range(h/size):
        for j in range(w/size):
            output.append((np.sum(scale*img[size*i:size*i+size,size*j:size*j+size]),(size*i,size*j)))
    return sorted(output,key=lambda x:x[0],reverse=True)
# For each data subset
for datadir in DATA_SUBSETS:

    features = []
    labels = []
    paths = []
    images_list = glob.glob(datadir + "/*/*.jpg")
    print len(images_list)
    # Process images
    for index, path in enumerate(images_list):
        try:
            looks = np.zeros((9,2048))
            # Load image
            im = skimage.io.imread(path)
            im = helper.preprocess(im)
            # im = skimage.transform.resize(im, (224, 224), mode='constant').astype(np.float32)
            # im = np.expand_dims(im, axis=0)*255

            a = skimage.filters.gaussian(im, sigma=10, multichannel=True)
            s_path = "/home/csweeney/images_test/saliency/"+path.split("/")[-1]
            if im is None or not os.path.isfile(s_path): 
                print("cant load")
                continue
            image = skimage.io.imread(s_path)
            image = skimage.transform.resize(image,(244,244))
            size=75
            imgs = saliency_queue(image,size,scale =1)
            for i,j in enumerate(imgs[:9]):
                a[j[1][0]:j[1][0]+size,j[1][1]:j[1][1]+size] = im[j[1][0]:j[1][0]+size,j[1][1]:j[1][1]+size]          
                looks[i,:] = features_model.predict(a).flatten()

            # Cache result
            label = NAMES_TO_IDS[os.path.basename(os.path.dirname(path))]
            labels.append(label)
            features.append(looks)
            rel_path = os.path.join(*path.split(os.sep)[-2:]).replace("\\", "/")
            paths.append(rel_path)

            # Show progress
            if index % 100 == 0:
                print(index, "/", len(images_list))

        except Exception as e:
            print("Error processing path {}: {}".format(path, e))

    # Save to disk
    np.save(os.path.join(datadir, FEATURES_FILENAME), features)
    np.save(os.path.join(datadir, LABELS_FILENAME), np.uint8(labels))
    with open(os.path.join(datadir, PATHS_FILENAME), mode='w') as f:
        json.dump(paths, f, indent=4)

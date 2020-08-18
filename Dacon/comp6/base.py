# train_test_fastai_models.py #
from fastai.vision.learner import *
from fastai.basic_train import *
from fastai.callbacks import *
from fastai.vision import *
from fastai import *
import fastai
import torch.nn as nn
import torch
import sklearn.metrics as skmetrics
import itertools
import copy
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import pickle
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import sys
import os
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


def training(subset):
    train_df = pd.read_csv("./data/"+subset+"train.csv",
                           usecols=["file", "label"])
    # valid_df = pd.read_csv("./data/"+subset+"_valid.csv",
    #                        usecols=["file", "label"])
    train_valid_df = train_df
    ds_tfms = get_transforms(
        do_flip=False, flip_vert=False, max_rotate=30, max_zoom=1.1, max_warp=0.3)
    # ImageDataBunch.from_folder seems to be dysfunctional
    data = ImageDataBunch.from_df(
        "", train_valid_df, ds_tfms=ds_tfms, size=28, bs=1024)
    print(data)
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.lr_find(start_lr=1e-6, end_lr=1e1, stop_div=True, num_it=200)

    plt.figure()
    learn.recorder.plot(suggestion=True)
    plt.title("Optimal Learning Rate - "+subset)
    plt.savefig("./graphics/"+subset+"_lr_selection.png")
    plt.savefig("./graphics/"+subset+"_lr_selection.pdf")
    plt.close()

    lr = learn.recorder.min_grad_lr
    print(lr)
    learn.fit_one_cycle(10, lr)

    plt.figure()
    learn.recorder.plot_metrics()
    plt.title("Training accuracies - "+subset)
    plt.savefig("./graphics/"+subset+"_train_accs.png")
    plt.savefig("./graphics/"+subset+"_train_accs.pdf")
    plt.close()

    plt.figure()
    learn.recorder.plot_lr()
    plt.title("Training Learning Rates - "+subset)
    plt.savefig("./graphics/"+subset+"_train_lr.png")
    plt.savefig("./graphics/"+subset+"_train_lr.pdf")
    plt.close()

    plt.figure()
    learn.recorder.plot_losses()
    plt.title("Training Losses - "+subset)
    plt.savefig("./graphics/"+subset+"_train_losses.png")
    plt.savefig("./graphics/"+subset+"_train_losses.pdf")
    plt.close()

    learn.export(subset+".pkl")


def testing(subset):
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.

        Original source: scikit-learn documentation
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = title+" (normalized) \n"
        else:
            title = title+"\n"

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    test = pd.read_csv("./data/"+subset+"_test.csv", usecols=["file", "label"])
    y_true = test["label"].values
    learn = load_learner("./", subset+".pkl",
                         test=ImageList.from_df(test, path=""))
    y_pred, _ = learn.get_preds(ds_type=DatasetType.Test)
    y_pred = np.argmax(y_pred.numpy(), axis=1)
    acc = skmetrics.accuracy_score(y_true, y_pred)
    title = subset+" Acc:"+str(np.round(acc, 4))+" confusion matrix"
    classes = sorted(np.unique(y_true))

    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(skmetrics.confusion_matrix(
        y_true=y_true, y_pred=y_pred), classes, title=title)
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(skmetrics.confusion_matrix(
        y_true=y_true, y_pred=y_pred), classes, title=title, normalize=True)
    plt.savefig("./graphics/"+subset+"_confusion_matrix.png")
    plt.savefig("./graphics/"+subset+"_confusion_matrix.pdf")
    plt.close()


def main():
    tick = time.time()
    print("fastai version:", fastai.__version__)
    print("PyTorch version:", torch.__version__)

    subsets = ["dacon/comp6/"]

    for subset in subsets:
        tick1 = time.time()
        print("Running fastai on", subset)
        training(subset)
        gc.collect()
        testing(subset)
        print("Subset finished after:", time.time()-tick, "seconds")
        gc.collect()
    print("Finished after:", (time.time()-tick)/3600, "hours")


if __name__ == "__main__":
    main()

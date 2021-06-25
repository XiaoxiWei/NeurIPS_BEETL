import matplotlib.pyplot as plt 
import numpy as np
import mne
import math
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from mne.filter import filter_data

import pickle
import os
import sys
from scipy import signal
import time
from sklearn.utils import shuffle

def balance_set(X,y,replace=False):
    labels= np.unique(y)
    labelsize=labels.shape[0]
    #print('labelsize:',labelsize)
    label_count = np.zeros(labelsize).astype(int)
    for i in range(labelsize):
        tempy = y[y==labels[i]]
        label_count[i]=y[y==labels[i]].shape[0]
    maxsize = label_count.max()
    for i in range(labelsize):
        tempy = y[y==labels[i]]
        tempx = X[y==labels[i]]
        tempx,tempy,ratio=get_sourceset_from_newraces(tempx,tempy,maxsize,random=True,replace=replace)
        if i ==0:
            balanced_data = tempx
            balanced_label = tempy
        else:
            balanced_data = np.concatenate((balanced_data,tempx))
            balanced_label = np.concatenate((balanced_label,tempy))
    return shuffle(balanced_data,balanced_label)
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
#     # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = signal.butter(order, [low, high], btype='bandpass')
    y= signal.lfilter(i, u, data)
    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    i, u = signal.butter(order, [low, high], btype='bandstop')
    y= signal.lfilter(i, u, data)
    return y


def lowpass_cnt(data,lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    i, u = signal.butter(order, low, btype='lowpass')
    y= signal.lfilter(i, u, data)
    return y

def highpass_cnt(data,highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    i, u = signal.butter(order, high, btype='highpass')
    y= signal.lfilter(i, u, data)
    return y


def filter_notch(data,notchcut,fs):
    f0 = notchcut  # Frequency to be removed from signal (Hz)
    w0 = f0 / (fs / 2)  # Normalized Frequency
    Q= 30
    i, u = signal.iirnotch(w0, Q)
#     data = signal.filtfilt(b, a, data)
    data= signal.lfilter(i, u, data)
    return(data)
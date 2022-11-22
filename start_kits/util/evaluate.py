#!/usr/bin/env python
import sys
import os
import os.path
import numpy as np

def get_label_weight(y):
    #print('start assessing')
    overall_counts = y.shape[0]
    label_amount = []
    weights_norm = []
    labels = np.unique(y)
    label_counts = labels.shape[0]
    #print(labels)
    for i in range(label_counts):
        temp = y[y==i]
        #print(temp.shape[0])
        label_amount.append(temp.shape[0])
        tempweight = float(0)
        tempweight = 100/(float(label_counts)*float(temp.shape[0]))
        # raise ValueError(
        #         "1: {}\n2:{}\n 3{}".format(label_counts,temp.shape[0],tempweight))
        #print(tempweight)
        weights_norm.append(tempweight)
    weights_norm = np.array(weights_norm)
    return weights_norm,label_counts,label_amount
def scoring(predict,truth):
    weights_norm,label_counts,label_amount = get_label_weight(truth)
    # print(weights_norm)
    
    score = float(0)
    if predict.shape != truth.shape:
        return None
    for label_iter in range(truth.shape[0]):
        true_label = truth[label_iter]
        pred_label = predict[label_iter]
        weight = weights_norm[true_label]
        if true_label == pred_label:
            score = score+weight
    return np.around(score,2)

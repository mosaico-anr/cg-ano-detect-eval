# Copyright (c) 2022 Orange - All rights reserved
# 
# Author:  Joël Roman Ky
# This code is distributed under the terms and conditions of the MIT License (https://opensource.org/licenses/MIT)
# 

from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np



def performance_evaluations(y_true, y_pred, y_score=None, 
                            return_values=False, plot_roc_curve=True, verbose=True):
    """The performance evaluation function.

    Args:
        y_true (np.array)               : The true labels
        y_pred (np.array)               : The predicted labels
        y_score (np.array, optional)    : The reconstruction score. Defaults to None.
        return_values (bool, optional)  : If the score must be returned. Defaults to False.
        plot_roc_curve (bool, optional) : If the ROC curve must be plotted. Defaults to True.
        verbose (bool, optional)        : Defaults to True.

    Returns:
            The Precision, Recall, F1-Score, AUC.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp/(tp+fp)
    recall = tp /(tp+fn)
    accuracy = (tp+tn)/(tp+tn+fn+fp)
    f1 = 2*precision*recall / (precision + recall)
    #print(precision, recall)
    
    fpr, tpr, _ = roc_curve(y_true,y_score)
    auc_val = roc_auc_score(y_true,y_score, labels=[0,1])
    
    #average_precision = average_precision_score(y_true, y_score)
    #print(average_precision)
    
    # Data to plot precision - recall curve
    #precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # Use AUC function to calculate the area under the curve of precision recall curve
    #aupr = auc(recalls, precisions)
    #print(aupr)
    
    if verbose:
        print(f"Performance evaluation :\nPrecision = {precision:.2f}\nRecall = {recall:.2f}\nAccuracy = {accuracy:.2f}\nF1-Score = {f1:.2f}\nAUC = {auc_val:.2f}\n")
        #print(f"AP = {average_precision:.2f}\nAUPR = {aupr:.2f}\n")
    
    if plot_roc_curve:        

        #idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()
        plt.figure()
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(fpr,tpr,label=f"AUC = {auc_val:.2f}", color="darkorange", lw=2)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        #plt.plot(fpr[idx],tpr[idx], 'ro')
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()
    if return_values:
        return precision, recall, f1, auc_val
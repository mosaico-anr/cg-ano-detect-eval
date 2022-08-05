from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit



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
    

def threshold_optimization(test_score, y_true, number_pertiles=100, val_ratio=.2, seed=42):
    """The threshold optimization function.

    Args:
        test_score (np.array)           : The test score.
        y_true (np.array)               : The true labels.
        number_pertiles (int, optional) : The number of percentiles. Defaults to 100.
        val_ratio (float, optional)     : The ratio of the test set to use 
                                          for threshold optimization. Defaults to .2.
        seed (int, optional)            : The random generator seed. Defaults to 42.

    Returns:
            The precision, recall, f1, auc got on the (1-val_ratio) of the test set.
    """
    # Generate indices the testscore
    #n = len(test_score)
    #idx = list(range(n))
    #shuffle(idx)
    #idx = np.array(idx)

    # split score in test and validation
    #n_test = int(n * (1 - val_ratio))
    #print(type(test_score))
    
    y_true_arr = np.array(y_true)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    for t_index, v_index in split.split(test_score, y_true):
        score_t, y_t = test_score[t_index], y_true_arr[t_index]
        score_v, y_v = test_score[v_index], y_true_arr[v_index]
        
    #print(pd.Series(y_v).value_counts())
    #print(pd.Series(y_t).value_counts())
    
    #print(y_true_arr)
    #print(len(y_true_arr), len(test_score))
    
    #score_t = test_score[idx[:n_test]]
    #y_t = y_true_arr[idx[:n_test]]
    #score_v = test_score[idx[n_test:]]
    #y_v = y_true_arr[idx[n_test:]]

    # Estimate the threshold on the validation set
    threshold = get_best_f1_threshold(score_v, y_v, number_pertiles)
    
    y_pred = (score_t >= threshold).astype(int)
    #y_pred = np.array(list(map(lambda val : 0 if val else 1, y_pred)))
    
    
    # Compute metrics on the test set
    precision, recall, f1, auc = performance_evaluations(y_true=y_t, 
                                                     y_pred=y_pred, 
                                                     y_score=score_t, 
                                                     return_values=True, plot_roc_curve=False)

    return precision, recall, f1, auc

def get_best_f1_threshold(test_score, y_true, number_pertiles, verbose=True):
    """The threshold optimization on F1-score.

    Args:
        test_score (np.array)   : The test score.
        y_true (np.array)       : The true labels.
        number_pertiles (int)   : The number of percentiles.
        verbose (bool, optional): . Defaults to True.

    Returns:
            The threshold that yields the best F1-score.
    """
    ratio = 100 * sum(y_true == 0) / len(y_true)
    print(f"Ratio of normal data:{ratio:.2f}")
    q = np.linspace(max(ratio - 5, 0), min(ratio + 5, 100), number_pertiles)
    thresholds = np.percentile(test_score, q)

    f1 = np.zeros(shape=number_pertiles)
    r = np.zeros(shape=number_pertiles)
    p = np.zeros(shape=number_pertiles)


    for i, (thresh, _) in enumerate(zip(thresholds, q)):
        
        y_pred = (test_score >= thresh).astype(int)

        #print(y_pred)
        #print(y_true)
        #unique_values = np.union1d(y_true, y_pred)
        #print((unique_values))
        
        _, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        p[i] = tp/(tp+fp)
        r[i] = tp /(tp+fn)
        f1[i] = 2*p[i]*r[i] / (p[i] + r[i])

    arm = np.argmax(f1)
    if verbose:
        print(f"Best metrics are :\tPrecision = {p[arm]:.2f}\tRecall = {r[arm]:.2f}\tF1-Score = {f1[arm]:.2f}\n")
    return thresholds[arm]
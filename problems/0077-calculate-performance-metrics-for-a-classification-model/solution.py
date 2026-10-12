import numpy as np
from collections import Counter

def confusion_matrix(data):
    counts = Counter(zip(*data))  # pairs up (actual, predicted)
    tp = counts[(1,1)]
    fn = counts[(1,0)]
    fp = counts[(0,1)]
    tn = counts[(0,0)]
    tp_fn = [tp, fn]
    fp_tn = [fp, tn]
    confusion_matr = [tp_fn, fp_tn]
    return confusion_matr

def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    incorrects = np.sum(np.abs(y_true - y_pred))
    accuracy = 1-incorrects/len(y_true)
    return accuracy

def precision(y_true, y_pred):
    # precision metric  = tp / (tp + fp)
    # out of all the preds flagged as +ve how many of them were correct
    positive_pred = 0 # tp + fp
    tp = 0
    for i in range(len(y_true)):
        if y_pred[i]==1:
            positive_pred +=1
        if y_pred[i] == 1 and y_true[i]==1:
            tp += 1
    
    if positive_pred == 0:
        return 0
    return tp/positive_pred

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # recall = tp / (tp + fn) -> how many were tp from the labels that should have been predicted positives
    actual_positives  = 0 
    fn = 0
    tp = 0
    for i in range(len(y_true)):
        if y_true[i]==1:
            actual_positives +=1
        if y_pred[i] == 0 and y_true[i]==1:
            fn += 1
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
    if actual_positives == 0:
        return 0
    return tp/actual_positives


def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:

    data = [actual, predicted]
    confusion_mat = confusion_matrix(data)
    accuracy = accuracy_score(actual, predicted)
    pres = precision(actual, predicted)
    rec = recall(actual, predicted)
    f1 = 2*(pres*rec)/(pres+rec) if (pres + rec) != 0 else 0

    # specificity -> how well are we labelling the negative elements correctly ? -> sp = tn / (tn+fp)
    tn, fp = confusion_mat[1][1], confusion_mat[1][0]
    fn = confusion_mat[0][1]
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    #negative predictive value -> how many elements labeled as negative are actually negative ? -> tn/(tn+fn)

    negativePredictive = tn / (tn + fn) if (tn + fn) != 0 else 0


    return confusion_mat, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
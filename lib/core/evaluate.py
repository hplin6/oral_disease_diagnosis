# ------------------------------------------------------------------------------
# Copy from https://github.com/HRNet/HRNet-Image-Classification
# Modified by us
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from sklearn.metrics import roc_curve, auc,confusion_matrix,multilabel_confusion_matrix,classification_report
from sklearn.metrics import average_precision_score,precision_recall_curve,recall_score,precision_score,f1_score,accuracy_score,roc_auc_score
import matplotlib.pyplot as plt

import math
import numpy as np
import sklearn.metrics as skm

class PrintColors:
    GREEN = "\033[0;32m"
    BLUE = "\033[1;34m"
    RED = "\033[1;31m"

    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_COLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def c_statistic_with_95p_confidence_interval(cstat, num_positives, num_negatives, z_alpha_2=1.96):
    """
    Calculates the confidence interval of an ROC curve (c-statistic), using the method described
    under "Confidence Interval for AUC" here:
      https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf
    Args:
        cstat: the c-statistic (equivalent to area under the ROC curve)
        num_positives: number of positive examples in the set.
        num_negatives: number of negative examples in the set.
        z_alpha_2 (optional): the critical value for an N% confidence interval, e.g., 1.96 for 95%,
            2.326 for 98%, 2.576 for 99%, etc.
    Returns:
        The 95% confidence interval half-width, e.g., the Y in X ± Y.
    """
    q1 = cstat / (2 - cstat)
    q2 = 2 * cstat ** 2 / (1 + cstat)
    numerator = cstat * (1 - cstat) \
                + (num_positives - 1) * (q1 - cstat ** 2) \
                + (num_negatives - 1) * (q2 - cstat ** 2)
    standard_error_auc = math.sqrt(numerator / (num_positives * num_negatives))
    return z_alpha_2 * standard_error_auc

def roc_auc(ground_truth, probs, index):
    gts = np.argmax(ground_truth, axis=1)
    n_gts = np.zeros_like(gts)
    n_gts[gts == index] = 1
    n_pos = np.sum(n_gts == 1)
    n_neg = n_gts.size - n_pos
    n_ps = probs[..., index]#.squeeze()
    n_gts, n_ps = n_gts.ravel(), n_ps.ravel()
    return n_pos, n_neg, skm.roc_auc_score(n_gts, n_ps)

def roc_auc_set(ground_truth, probs, index):
    gts = np.argmax(ground_truth, axis=1)
    max_ps = np.max(probs[..., index], axis=0)
    max_gts = np.any(gts == index, axis=0)
    pos = np.sum(max_gts)
    neg = max_gts.size - pos
    print(max_gts, max_ps)
    return pos, neg, skm.roc_auc_score(max_gts, max_ps)

def print_aucs(ground_truth, probs):
    seq_tauc = 0.0;
    seq_tot = 0.0
    set_tauc = 0.0;
    set_tot = 0.0
    print("\t AUC")
    for idx, cname in zip([0,1,2,3,4],["normal","ulcer","low","high","cancer"]):
        pos, neg, seq_auc = roc_auc(ground_truth, probs, idx)
        seq_tot += pos
        seq_tauc += pos * seq_auc
        seq_conf = c_statistic_with_95p_confidence_interval(seq_auc, pos, neg)
        #pos, neg, set_auc = roc_auc_set(ground_truth, probs, idx)
        #set_tot += pos
        #set_tauc += pos * set_auc
        #set_conf = c_statistic_with_95p_confidence_interval(set_auc, pos, neg)
        print("{: <8}\t{:.3f} ({:.3f}-{:.3f})".format(cname, seq_auc, seq_auc - seq_conf,seq_auc + seq_conf))
        #print("{: <8}\t{:.3f} ({:.3f}-{:.3f})\t{:.3f} ({:.3f}-{:.3f})".format(cname, seq_auc, seq_auc - seq_conf, seq_auc + seq_conf,
            #set_auc, set_auc - set_conf, set_auc + set_conf))
    print("Average\t\t{:.3f}".format(seq_tauc / seq_tot))
    #print("Average\t\t{:.3f}\t{:.3f}".format(seq_tauc / seq_tot, set_tauc / set_tot))

def print_confusion_matrix(cm, labels):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [12])
    # Print header
    print()
    first_cell = "True\Pred"
    print("|%{0}s|".format(columnwidth - 2) % first_cell, end="")
    for label in labels:
        print("%{0}s|".format(columnwidth -1) % label, end="")
    print()

    first_cell = "-------"
    print("|%{0}s|".format(columnwidth-2) % first_cell, end="")
    for _ in labels:
        print("%{0}s|".format(columnwidth-1) % first_cell, end="")
    print()

    # Print rows
    for i, label1 in enumerate(labels):
        print("|%{0}s|".format(columnwidth - 2) % label1, end="")
        for j in range(len(labels)):
            cell = "%{0}.2f|".format(columnwidth-1) % cm[i, j]
            if i == len(labels) - 1 or j == len(labels) - 1:
                cell = "%{0}d|".format(columnwidth-1) % cm[i, j]
                if i == j:
                    print("%{0}s|".format(columnwidth-1) % ' ', end="")
                else:
                    print(PrintColors.BLUE + cell + PrintColors.END_COLOR, end="")
            elif i == j:
                print(PrintColors.GREEN + cell + PrintColors.END_COLOR, end="")
            else:
                print(PrintColors.RED + cell + PrintColors.END_COLOR, end="")
        print()

def save_pregt_result(filename,gt_cls,pre_cls):
    output= np.zeros((2,gt_cls.shape[0]), dtype='uint8')
    output[0][:]=gt_cls
    output[1][:]=pre_cls
    np.savetxt(filename,output,delimiter=",")

def cal_confusion_matric(cm_mat,num_labels):
    TP=[0]*num_labels
    FN=[0]*num_labels
    FP=[0]*num_labels
    TN=[0]*num_labels
    ACC=[0]*num_labels
    TPR=[0]*num_labels
    PRECISION=[0]*num_labels
    TNR=[0]*num_labels
    FPR=[0]*num_labels
    F1_score=[0]*num_labels
    for i in range(num_labels):
        TP[i] = np.sum(cm_mat[i,i])
        FP[i] = np.sum(cm_mat, axis=0)[i] - TP[i]
        FN[i] = np.sum(cm_mat, axis=1)[i] - TP[i]
        TN[i] = np.sum(cm_mat) - TP[i] - FP[i] - FN[i]
        #print(TP[i],FP[i],FN[i],TN[i])
        ACC[i] = (TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i])
        TPR[i] = TP[i] / (TP[i] + FN[i])
        if TPR[i] is None:
            TPR[i] = 0

        PRECISION[i] = TP[i] / (TP[i] + FP[i])
        if PRECISION[i] is None:
            PRECISION[i] = 0

        TNR[i] = TN[i] / (TN[i] + FP[i])
        if TNR[i] is None:
            TNR[i] = 0

        FPR[i] = FP[i] / (TN[i] + FP[i])
        if FPR[i] is None:
            FPR[i] = 0

        F1_score[i] = (2 * (PRECISION[i] * TPR[i])) / (PRECISION[i] + TPR[i])
        #P[i] = TP[i] + FN[i]
        #N[i] = FP[i] + TN[i]
    Accuracy= np.mean(ACC)
    Sensitivity = np.mean(TPR)
    Specificity = np.mean(TNR)
    Precision = np.mean(PRECISION)
    #FalsePositiveRate = np.mean(FPR)
    F1_score = np.mean(F1_score)

    return Accuracy,Sensitivity,Specificity,Precision,F1_score

def swith_classes(y_true, y_pred):
    #classes = ['cancer','highrisk,'lowrisk','normal',"ulcer"]
    #classes = ['normal','ulcer','lowrisk','highrisk',"cancer"]
    ty_true=np.zeros_like(y_true)
    ty_pred=np.zeros_like(y_pred)
    ty_pred[y_pred==0]=4
    ty_pred[y_pred==1]=3
    ty_pred[y_pred==2]=2
    ty_pred[y_pred==3]=0
    ty_pred[y_pred==4]=1

    ty_true[y_true == 0] = 4
    ty_true[y_true == 1] = 3
    ty_true[y_true == 2] = 2
    ty_true[y_true == 3] = 0
    ty_true[y_true == 4] = 1
    return ty_true,ty_pred

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def cal_metrics(gts,et_clss):
    gt_clss=np.zeros((len(gts),5), dtype=np.uint8)
    for idx,gt_cls in enumerate(gts):
        gt_clss[idx,gt_cls]=1

    et_array=np.zeros_like(gt_clss)
    for idx,et_cls in enumerate(et_clss):
        ccls = np.argmax(et_cls)
        et_array[idx,ccls]=1

    y_true=np.argmax(gt_clss, axis = 1)
    y_pred=np.argmax(et_array, axis = 1)
    y_true, y_pred=swith_classes(y_true, y_pred)
    cmat=confusion_matrix(y_true,y_pred)
    print(cmat)

    accuracy,sensitivity,specificity,precision,F1_score=cal_confusion_matric(cmat,5)
    print("@ Mean Acc=%.4f,Macro: Sensitivity=%.4f,specificity=%.4f,Precision=%.4f,F1_score=%.4f" %(accuracy,sensitivity,specificity,precision,F1_score))

    return accuracy,sensitivity,specificity,F1_score,y_true,y_pred,cmat

def get_confusion_matrix(cnf_mat, norm_cm=True, print_cm=True):

    class_names = ['normal', 'ulcer', 'lowrisk', 'highrisk', "cancer"]
    total_cnf_mat = np.zeros(shape=(cnf_mat.shape[0] + 1, cnf_mat.shape[1] + 1), dtype=np.float)
    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    for i_row in range(cnf_mat.shape[0]):
        total_cnf_mat[i_row, -1] = np.sum(total_cnf_mat[i_row, 0:-1])

    for i_col in range(cnf_mat.shape[1]):
        total_cnf_mat[-1, i_col] = np.sum(total_cnf_mat[0:-1, i_col])

    if norm_cm:
        cnf_mat = cnf_mat/(cnf_mat.astype(np.float).sum(axis=1)[:, np.newaxis] + 0.001)

    total_cnf_mat[0:cnf_mat.shape[0], 0:cnf_mat.shape[1]] = cnf_mat

    if print_cm:
        print_confusion_matrix(cm=total_cnf_mat, labels=class_names + ['TOTAL', ])

    return cnf_mat

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        batch_size = target.size(0)
        correct = correct*(100.0 / batch_size)
        return correct

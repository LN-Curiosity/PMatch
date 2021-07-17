#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/7/8 0008 19:04
# @Author : zhe lang
# @Site : 
# @File : threshold_eval.py
# @Software:


import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def eval_thr():
    with open('./roc/prediction_result_sif.txt-cross', 'r') as p_fp1:
        y_score1 = list(p_fp1)
    with open('./roc/label_result_sif.txt-cross', 'r') as l_fp1:
        y_label1 = list(l_fp1)
    with open('./roc/prediction_result_innereye.txt-cross', 'r') as p_fp2:
        y_score2 = list(p_fp2)
    with open('./roc/label_result_innereye.txt-cross', 'r') as l_fp2:
        y_label2 = list(l_fp2)

    y_score1 = [float(score.strip()) for score in y_score1]
    y_label1 = [float(label.strip()) for label in y_label1]
    y_score1 = np.array(y_score1)
    y_label1 = np.array(y_label1)
    y_score2 = [float(score.strip()) for score in y_score2]
    y_label2 = [float(label.strip()) for label in y_label2]
    y_score2 = np.array(y_score2)
    y_label2 = np.array(y_label2)

    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_label1, y_score1)
    roc_auc1 = metrics.auc(fpr1, tpr1)
    print(roc_auc1)
    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_label2, y_score2)
    roc_auc2 = metrics.auc(fpr2, tpr2)
    print(roc_auc2)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr1, tpr1, color='red', linestyle='--', linewidth=1.5, label='PMatch, AUC = %0.3f' % roc_auc1)
    plt.plot(fpr2, tpr2, color='blue', linestyle='--', linewidth=1.5, label='InnerEye, AUC = %0.3f' % roc_auc2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc="lower right", fontsize=25)
    plt.tight_layout()
    plt.savefig('./roc-cross.png')
    plt.show()

    maxindex1 = (tpr1 - fpr1).tolist().index(max(tpr1 - fpr1))
    best_thr1 = thresholds1[maxindex1]
    maxindex2 = (tpr2 - fpr2).tolist().index(max(tpr2 - fpr2))
    best_thr2 = thresholds2[maxindex2]
    print(best_thr1)
    print(best_thr2)


if __name__ == '__main__':
    eval_thr()

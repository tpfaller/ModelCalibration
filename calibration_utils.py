#! venv/bin/python
import time
import itertools
from itertools import cycle
from matplotlib import pyplot
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

import numpy as np

import pandas as pd
import math

from sklearn.metrics import roc_curve, auc,  precision_recall_curve, average_precision_score, matthews_corrcoef
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score, classification_report, log_loss, confusion_matrix, accuracy_score 
from sklearn.utils import class_weight
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
# import seaborn as sns
# import ml_insights as mli
# from betacal import BetaCalibration

#custom functions to generate reliability diagram
def calc_bins(y_test, preds):
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)  
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (y_test[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]
        else:
            bin_accs[bin] = 0.
            bin_confs[bin] = 0.

    return bins, binned, bin_accs, bin_confs, bin_sizes


# get ECE and MCE metrics

def get_metrics(y_test, preds):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_test, preds)
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)
    return ECE,MCE



def draw_reliability_graph(y_test, preds):
    ECE, MCE = get_metrics(y_test, preds)
    bins, _, bin_accs, _, _ = calc_bins(y_test, preds)
    fig = plt.figure(figsize=(15, 10), dpi=400)
    ax = fig.gca()
    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    # x/y labels
    plt.xlabel('Prediction', fontsize=20)
    plt.ylabel('Truth', fontsize=20)
    # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')
    # Error bars
    plt.bar(bins, bins,width=0.1,alpha=0.3,edgecolor='black', color='orange', hatch='\\')
    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, 
          edgecolor='black', color='red') # b before 
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)
    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')
    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='blue',  #green before
                             label='ECE = {:.2f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='green', 
                                label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch],prop={'size': 15})  

# Plot confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# compute performance metrics

def matrix_metrix(real_values,pred_values,beta):
    CM = confusion_matrix(real_values,pred_values)
    TN = CM[0][0]
    FN = CM[1][0] 
    TP = CM[1][1]
    FP = CM[0][1]
    Population = TN+FN+TP+FP
    Kappa = 2 * (TP * TN - FN * FP) / (TP * FN + TP * FP + 2 * TP * TN + FN**2 + FN * TN + FP**2 + FP * TN)
    Prevalence = round( (TP+FP) / Population,2)
    Accuracy   = round( (TP+TN) / Population,4)
    Precision  = round( TP / (TP+FP),4 )
    NPV        = round( TN / (TN+FN),4 )
    FDR        = round( FP / (TP+FP),4 )
    FOR        = round( FN / (TN+FN),4 ) 
    check_Pos  = Precision + FDR
    check_Neg  = NPV + FOR
    Recall     = round( TP / (TP+FN),4 )
    FPR        = round( FP / (TN+FP),4 )
    FNR        = round( FN / (TP+FN),4 )
    TNR        = round( TN / (TN+FP),4 ) 
    check_Pos2 = Recall + FNR
    check_Neg2 = FPR + TNR
    LRPos      = round( Recall/FPR,4 ) 
    LRNeg      = round( FNR / TNR ,4 )
    DOR        = round( LRPos/LRNeg)
    F1         = round ( 2 * ((Precision*Recall)/(Precision+Recall)),4)
    FBeta      = round ( (1+beta**2)*((Precision*Recall)/((beta**2 * Precision)+ Recall)) ,4)
    MCC        = round ( ((TP*TN)-(FP*FN))/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))  ,4)
    BM         = Recall+TNR-1
    MK         = Precision+NPV-1
    mat_met = pd.DataFrame({'Metric':['TP','TN','FP','FN','Prevalence','Accuracy','Precision','NPV','FDR','FOR','check_Pos','check_Neg','Recall','FPR','FNR','TNR','check_Pos2','check_Neg2','LR+','LR-','DOR','F1','FBeta','MCC','BM','MK','Kappa'],     'Value':[TP,TN,FP,FN,Prevalence,Accuracy,Precision,NPV,FDR,FOR,check_Pos,check_Neg,Recall,FPR,FNR,TNR,check_Pos2,check_Neg2,LRPos,LRNeg,DOR,F1,FBeta,MCC,BM,MK, Kappa]})
    return (mat_met)


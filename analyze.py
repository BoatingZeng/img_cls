from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


def confuse_matrix(d, threshold):
    print('threshold：{0}'.format(threshold))
    total = d.shape[0]
    print('TOTAL：{0}'.format(total))
    d.loc[d['true_class'] != 0, 'true_class'] = 1
    d['pred'] = 0
    d.loc[d['ill'] > threshold, 'pred'] = 1

    tp = d[(d['true_class'] == 1) & (d['pred'] == 1)].shape[0]
    tn = d[(d['true_class'] == 0) & (d['pred'] == 0)].shape[0]
    fp = d[(d['true_class'] == 0) & (d['pred'] == 1)].shape[0]
    fn = d[(d['true_class'] == 1) & (d['pred'] == 0)].shape[0]

    print('TP：{0}'.format(tp))
    print('TN：{0}'.format(tn))
    print('FP：{0}'.format(fp))
    print('FN：{0}'.format(fn))

    assert total == (tp+tn+fp+fn)

    print('整体准确率：(TP+TN)/TOTAL = {0}'.format((tp + tn) / total))
    print('正例准确率：TP/(TP+FP) = {0}'.format(tp / (tp + fp)))
    print('正例召回率(灵敏度)：TP/(TP+FN) = {0}'.format(tp / (tp + fn)))
    print('误诊率：FP/(FP+TN) = {0}'.format(fp / (fp + tn)))


def roc(d):
    d.loc[d['true_class'] != 0, 'true_class'] = 1

    y_true = d['true_class']
    y_score = d['ill']

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('auc：{0}'.format(roc_auc))

    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def prc(d):
    d.loc[d['true_class'] != 0, 'true_class'] = 1

    y_true = d['true_class']
    y_score = d['ill']

    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=1)

    plt.plot(recall, precision, label='PRC curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()


def roc_multi(d, is_save_data=False):
    n_classes = 4
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'deeppink']
    enc = OneHotEncoder(sparse=False)
    y_true = d[['true_class']].values
    y_true -= 1
    enc.fit(y_true)
    y_true = enc.transform(y_true)
    y_score = d[['1', '2', '3', '4']].values

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    threshold = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], threshold[i] = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        if is_save_data:
            df = pd.DataFrame({'fpr': fpr[i], 'tpr': tpr[i], 'threshold': threshold[i]})
            df.to_csv('val_4_cls_{0}_roc_data.csv'.format(i+1), index=False)

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i+1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

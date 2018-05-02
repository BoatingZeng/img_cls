from sklearn import metrics
import matplotlib.pyplot as plt


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

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print('auc：{0}'.format(roc_auc))

    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def prc(d):
    d.loc[d['true_class'] != 0, 'true_class'] = 1

    y_true = d['true_class']
    y_score = d['ill']

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score, pos_label=1)

    plt.plot(recall, precision, label='PRC curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()

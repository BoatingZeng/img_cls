# 一些链接
* 代码地址：https://github.com/BoatingZeng/img_cls
* kaggle比赛地址：https://www.kaggle.com/c/diabetic-retinopathy-detection
* 结果数据示例可以直接看example_data目录

# 二分类结果

## 说明
误诊率(False Positive Rate)：在没有病的人当中，有多少被说成有病。

正例准确率(Precision)：说你是有病，你有多大概率是有病。(反过来可以知道，说你是有病，你有多大概率其实没病)

仔细对比上面两个概念。这两个都是条件概率，误诊率是条件于真实情况的，而Precision是条件于你的预测的。

ROC曲线的横坐标是误诊率，纵坐标是灵敏度(True Positive Rate)(召回率)。

PRC曲线的横坐标是召回率，纵坐标是准确率。

## 分析
备注：如果知道**ROC曲线**和**PRC曲线**的概念，就不用看下面的。用sklearn可以直接画这两个曲线。

模型输出的是两个score(二分类，两个score和为1)，只观察判断有病score。设置不同阀值，当有病score高于阀值，就判断为有病。

模型Performance(正例准确率(Precision)、(灵敏度)召回率、误诊率)随着阀值变化而变化，ROC曲线和PRC曲线就是描述这种变化关系。

* PRC曲线描述准确率和召回率之间的关系，参考：https://www.zhihu.com/question/30643044/answer/224360465
* ROC曲线描述召回率和误诊率之间的关系，参考：https://www.deeplearn.me/1522.html

### 混淆矩阵示例
总数(TOTAL)：6954

**阀值(threshold)：0.5**

table   |预测有病|预测无病|
--------|----------|--------
实则有病|  1227(TP)|  601(FN)
实则无病|   181(FP)|  4945(TN)

整体准确率(Accuracy)：(TP+TN)/TOTAL = 0.8875

正例准确率(Precision)：TP/(TP+FP) = 0.8714
灵敏度(召回率)：TP/(TP+FN) = 0.6712
误诊率：FP/(FP+TN) = 0.0353


**阀值(threshold)：0.4**

table   |预测有病|预测无病|
--------|----------|--------
实则有病|  1301(TP)|  527(FN)
实则无病|   273(FP)|  4853(TN)

整体准确率(Accuracy)：(TP+TN)/TOTAL = 0.8849

正例准确率(Precision)：TP/(TP+FP) = 0.8265
灵敏度(召回率)：TP/(TP+FN) = 0.7117
误诊率：FP/(FP+TN) = 0.0353


**阀值(threshold)：0.35**

table   |预测有病|预测无病|
--------|----------|--------
实则有病|  1333(TP)|  495(FN)
实则无病|   328(FP)|  4798(TN)

整体准确率(Accuracy)：(TP+TN)/TOTAL = 0.8816

正例准确率(Precision)：TP/(TP+FP) = 0.8025
灵敏度(召回率)：TP/(TP+FN) = 0.7292
误诊率：FP/(FP+TN) = 0.0639
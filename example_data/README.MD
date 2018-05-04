# 示例数据说明

* vgg16_2_cls_val_prob.csv：验证集上的预测结果
    * image：图片名字，不带后缀
    * not_ill：该例是没病的分数(概率，score)
    * ill：该例是有病的分数(概率，score)。not_ill + ill == 1。
    * true_class：该例的真实情况。0：没病；5：有病。
* val_roc_data.csv：用`sklearn.metrics.roc_curve`生成的数据，用来画roc曲线(验证集上的结果)
* val_roc_curve.png：画出来的roc曲线，那个area就是AUC值(验证集上的结果)
* val_prc_data.csv：用`sklearn.metrics.precision_recall_curve`生成的数据，用来画prc曲线(验证集上的结果)
* val_prc_curve.png：画出来的prc曲线(验证集上的结果)
* test_prc_curve.png：测试集上的prc曲线
* test_roc_curve.png：测试集上的roc曲线
* trainLabels.csv：比赛官方给出的训练集的label
* testLabels.csv：比赛后有人把测试集的标准答案放出来了，来源不明，经过测试(上传这个答案可以拿满分)，是正确的

# 数据的处理示例代码

```python
import pandas as pd
from analyze import confuse_matrix, roc, prc  # analyze是项目下的analyze.py文件

# 读取数据
data = pd.read_csv('example_data/vgg16_2_cls_val_prob.csv')
# 计算混淆矩阵
confuse_matrix(data,threshold=0.5)
# 画roc和prc
roc(data)
prc(data)
```

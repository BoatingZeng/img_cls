# 一些链接
* 代码地址：https://github.com/BoatingZeng/img_cls
* kaggle比赛地址(糖网检测)：https://www.kaggle.com/c/diabetic-retinopathy-detection
* kaggle比赛地址(司机行为检测)：https://www.kaggle.com/c/state-farm-distracted-driver-detection
* 结果数据示例可以直接看example_data目录

**下面的分析都是针对糖网检测的，并且记录的分析过程是针对验证集(验证集是从官方训练集每个类随机抽20%出来产生的)**

# 训练前的图像处理
参考kaggle比赛第一名的方法：https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801

主要是在他的基础上添加了截取图片中间正方形，最后得到的图片都是正方形的，后面所有训练分析都是基于这些图片，不使用原图，最终使用的代码如下：

```python
# src：源目录，des：生成图片保存目录，scale：图片缩放系数
def preprocess_img(src, des, scale=300):
    if len(os.listdir(des)) != 0:
        print('{0} is not empty'.format(des))
        return

    classes = os.listdir(src)

    for cls in classes:
        old_class_dir = os.path.join(src, cls)
        new_class_dir = os.path.join(des, cls)
        os.mkdir(new_class_dir)
        for f in glob.glob(os.path.join(old_class_dir, '*.jpeg')):
            try:
                a = cv2.imread(f)
                # scale img to a given radius
                a = scaleRadius(a, scale)
                # subtract local mean color
                a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale/30), -4, 128)
                # remove outer 10%
                b = np.zeros(a.shape)
                cv2.circle(b, (a.shape[1]//2, a.shape[0]//2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
                a = a*b+128*(1-b)
                # to square
                height = a.shape[0]
                width = a.shape[1]
                a = a[:, (width-height)//2:(width+height)//2, :]
                basename = os.path.basename(f)
                newpath = os.path.join(new_class_dir, basename)
                cv2.imwrite(newpath, a)
            except:
                print(f)
```

# 输入神经网络时的图像处理

看keras的api：https://keras.io/preprocessing/image/

训练集(参与梯度下降的部分)做随机扭曲变形，验证集或者测试时用的图片不做这个操作

特别说明的参数，这个两个参数都是一种正则化操作，都是让输入数值保持在0附近：

* rescale=1. / 255：就是每个像素每个通道上的值都先除以255
* samplewise_center=True：对于一张图片，把它所有像素所有通道上的值求均值m，然后每个像素上每个通道的值减m

其他变形参数设置得比较随意

```python
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        samplewise_center=True,
        rotation_range=360,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(samplewise_center=True, rescale=1. / 255)
```

# CNN结构

这里只说明vgg16，并且基于keras。使用了keras提供的在imageNet上训练好的vgg16模型的参数，并且不要原本的输出层，把输出层替换如下：

这个是用来做分类的输出层：

```python
base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)

x = Flatten(name='output_flatten')(base_model.output)
x = Dense(256, activation='relu', name='output_fc_cls'+str(class_num))(x)
x = Dropout(0.5)(x)
x = Dense(class_num, activation='softmax', name='output_predictions_cls'+str(class_num))(x)

model = Model(inputs=base_model.input, outputs=x, name='vgg16_cls'+str(class_num))
```

这个是用来做回归的输出层：

```python
base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)

x = Flatten(name='output_flatten')(base_model.output)
x = Dense(256, activation='relu', name='output_fc_reg')(x)
x = Dropout(0.5)(x)
x = Dense(1, name='output_predictions_reg')(x)

model = Model(inputs=base_model.input, outputs=x, name='vgg16_reg')
```

详细请参考keras的使用方法和models.py里的代码，另外这个blog：https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

**注意的问题**

1. 训练时要先冻结输出层以外的参数，先单独训练输出层，这个操作是为了避免随机初始化的输出层参数在梯度下降时对前面的参数造成过大的破坏，至于训练多久，看情况，感觉模型分数没什么变化，就可以停了
2. 训练其他层，在这个代码中，是通过config里的train_layers设置的(参考train_vgg16_config.json.example文件)，在train_layers列表里的block是会被训练的(其余block会被冻结，输出层任何情况下都会被训练)。block的含义要看vgg16的模型描述，可以参考keras代码：https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
3. 经过测试，猫狗辨别问题和开头提到的司机行为检测问题，只要训练block5(输出层是必须训练的)就可以得到不错的结果(准确率95%)，但是糖网检测只训练block5结果不理想，所以直接训练全部。当然，初始参数都是基于imageNet训练出来的参数的

# 样本不平衡的处理
使用keras的class_weight，做分类时，class_weight设置为每个类所占比例的反比，比如有病没病二分类，训练集里无病(0)样本20,682，有病(5)样本7,488，那么0类和5类的class_weight分别设置为1.0和3.0

# 有病没病二分类结果
把有病的1、2、3、4类放在一起，当作一类，然后和没病的0一起做二分类

## 说明
* 误诊率(False Positive Rate)：在没有病的人当中，有多少被说成有病。

* 正例准确率(Precision)：说你是有病，你有多大概率是有病。(反过来可以知道，说你是有病，你有多大概率其实没病)
* 召唤率(灵敏度)：就是对一个有病的群体，能把多少病人找出来

对比上面三个概念。这三个都是条件概率，误诊率和召回率都是条件于真实情况的，而Precision是条件于你的预测的。

* ROC曲线的横坐标是误诊率，纵坐标是灵敏度(True Positive Rate)(召回率)。

* PRC曲线的横坐标是召回率，纵坐标是正例准确率(Precision)。

## 分析
备注：如果知道**ROC曲线**和**PRC曲线**的概念，就不用看下面的。用sklearn可以直接画这两个曲线。

模型输出的是两个score(二分类，没病和有病，两个score和为1)，只观察判断有病score。设置不同阀值，当有病score高于阀值，就判断为有病。

模型Performance(正例准确率(Precision)、(灵敏度)召回率、误诊率)随着阀值变化而变化，ROC曲线和PRC曲线就是描述这种变化关系。可以通过ROC和PRC找出一个合适的阀值(threshold)来作为判断是否有病，至于什么是合适，这里不讨论。

* PRC曲线描述准确率和召回率之间的关系，参考：https://www.zhihu.com/question/30643044/answer/224360465
* ROC曲线描述召回率和误诊率之间的关系，参考：https://www.deeplearn.me/1522.html

### 混淆矩阵示例
验证集样本总数(TOTAL)：6954

下面列出的**整体准确率**意义不大

**阀值(threshold)：0.5**

table   |预测有病|预测无病|
--------|----------|--------
实则有病|  1227(TP)|  601(FN)
实则无病|   181(FP)|  4945(TN)

* 整体准确率(Accuracy)：(TP+TN)/TOTAL = 0.8875

* 正例准确率(Precision)：TP/(TP+FP) = 0.8714
* 灵敏度(召回率)：TP/(TP+FN) = 0.6712
* 误诊率：FP/(FP+TN) = 0.0353


**阀值(threshold)：0.4**

table   |预测有病|预测无病|
--------|----------|--------
实则有病|  1301(TP)|  527(FN)
实则无病|   273(FP)|  4853(TN)

* 整体准确率(Accuracy)：(TP+TN)/TOTAL = 0.8849

* 正例准确率(Precision)：TP/(TP+FP) = 0.8265
* 灵敏度(召回率)：TP/(TP+FN) = 0.7117
* 误诊率：FP/(FP+TN) = 0.0353


**阀值(threshold)：0.35**

table   |预测有病|预测无病|
--------|----------|--------
实则有病|  1333(TP)|  495(FN)
实则无病|   328(FP)|  4798(TN)

* 整体准确率(Accuracy)：(TP+TN)/TOTAL = 0.8816

* 正例准确率(Precision)：TP/(TP+FP) = 0.8025
* 灵敏度(召回率)：TP/(TP+FN) = 0.7292
* 误诊率：FP/(FP+TN) = 0.0639

# 四分类结果
对有病的案例进行训练，并且对有病的案例进行预测，区分1、2、3、4类。数字越大严重程度越高。

## 分析
多分类问题画roc曲线的方法跟二分类是一样的，就是单独对每个分类进行绘制。详细结果看example_data。

## 混淆矩阵
验证集样本数(TOTAL)：1828

**用argmax判断类别，把得分最高的类别作为预测类别**

table|预测1|预测2|预测3|预测4|总数
-----|-----|-----|-----|-----|---
真实1| 297 | 182 |  0  |  3  |482
真实2| 181 | 834 | 21  | 22  |1058
真实3|   8 |  89 | 54  |  6  |157
真实4|   0 |  42 | 13  | 76  |131
总数 | 486 | 1147| 88  | 107 |1828

* quadratic weighted kappa：0.62505(用`sklearn.metrics.cohen_kappa_score`计算)
* 各个类的误诊率(你不是这个类但是说你是这个类)：
    * 1：0.1404
    * 2：0.4064
    * 3：0.0203
    * 4：0.0182

## 实验

### 实验1
试试降低2类的误诊率，然后计算kappa系数

方法：我们保存了每个类的ROC数据，通过调整(提高)2类的threshold，来降低2类的误诊率

设置threshold=0.74301696，此时2类误诊率是fpr=0.1402，和1类的误诊率差不多了。

具体操作：

1. 凡是没到达threshold=0.74301696而被判断为2类的病例，都重新判定
2. 重新判定时，用其他三个类中得分最高(argmax)的类来给这个病例进行预测

实际操作时，如果某个实例的2类score<threshold，就设置这个2类score为-1，然后argmax

结果：
* 设置2类的threshold后，有551个样本需要重新分类
* 重新分类后kappa：0.6139(很遗憾，反而降低了)

### 实验2
维持2类的threshold=0.74301696不变，调整1类的threshold=0.69646007，此时1类的误诊率为fpr=0.049

操作过程和上面类似

结果：
* 比起最初单纯的argmax，现在有804个样本需要重新分类
* kappa：0.3046(更低)

# 二分类和四分类结合进行预测
这里针对官方的测试集进行预测

## 过程
1. 给二分类设置一个阀值threshold，有病的score高于threshold，就视为有病
2. 把有病的部分用四分类器打分，把这个病例定为得分最高的分类

把结果上传kaggle，不同threshold得分如下

threshold |Private Score|recall in val|accuracy
----------|-------------|-------------|--------
0.08794189|   0.58199   |0.90043      |
0.19595632|   0.71091   |0.80032      |0.77530
0.43068624|   0.75935   |0.70021      |0.83151
0.50000000|   0.75584   |0.67122      |0.83539

* threshold：二分类阀值
* Private Score：这个比赛的评价分数，评价准则是quadratic weighted kappa，因为我们有正确答案，所以可以用这个api来计算：`sklearn.metrics.cohen_kappa_score`
* recall in val：设置这个阀值时，在验证集中，正例(有病)的召回率
* accuracy：总体准确率，因为有测试集的正确答案，所以可以线下计算这个值

# resnet的尝试

## resnet50
网络结构完全没有修改，只把imagenet那个1000的输出层改成4，去训练四分类。包括输入图片大小在内的其他条件都跟上面的一样。resnet50是按照stage来命名网络的，分别是stage1到5。5是靠近输出层的部分。另外，训练时，整个网络的BN(batch normalization)层都是要训练的。训练用的显卡是1070ti(显存8G)，输入图片大小是512*512，batch size设置成4(设置成8就不行了)。

1. 只训练stage5，冻结前面的stage，这样训练出来，准确率只有63%左右
2. 如果训练stage4、stage5，就会出现明显的过拟合，训练集上可以到80%，但是验证集上只有60%，无意义
3. 整个网络训练，也是过拟合，无意义，分数和只训练stage4、stage5差不多

# xception的尝试
结构没修改，只把输出层换成4。同样是训练整个网络。和resnet50类似，会出现过拟合，没有训练到最后，不确定过拟合程度有多大。验证集的准确率60多，训练集70多。

# Leaky ReLU的测试
把原本vgg16的全部ReLU替换成Leaky ReLU。用原本已经训练好的weights以低学习率继续训练。

如果设置的alpha比较大(例如alpha=0.5)，就会出现最后softmax出来的结果，其中一个类是1.0，其他全部是0.0。

**二分类模型用alpha=0.001，阀值0.5**

table   |预测有病|预测无病|
--------|----------|--------
实则有病|  1193(TP)|  633(FN)
实则无病|   144(FP)|  4982(TN)

和前面的对比，准确率上升，但是召回降低。总体差不多。

ROC曲线的AUC=0.8936。也是差不多(之前0.898)。

以上这些差异，未必是Leaky ReLU带来的，因为就算用原本的ReLU继续训练原本的weights，也可能有浮动。

# 关于quadratic cohen kappa 和准确率(accuracy)的对比
之前猜测accuracy一定会比kappa高，但事实不然。取其中一个参赛者的结果，计算出的kappa=0.831，accuracy=0.805。

某参赛者结果：https://kaggle2.blob.core.windows.net/forum-message-attachments/88677/2796/2015_07_27_182759_25_log_mean_rank.csv.zip

另外，只考虑真实有病的样本并且预测有病的样本，该结果的accuracy=0.527，kappa=0.603。

考虑该结果的二分类性能(查出是否有病的性能)，recall=0.759，precision=0.839。(比我得到的二分类模型好很多)

**结论：二分类的性能才是关键**
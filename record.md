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
2. 训练其他层，在这个代码中，是通过config里的train_layers设置的(参考train_vgg16_config.json.example文件)，在train_layers列表里的block是会被训练的。block的含义要看vgg16的模型描述，可以参考keras代码：https://github.com/keras-team/keras/blob/master/keras/applications/vgg16.py
3. 经过测试，猫狗辨别问题和开头提到的司机行为检测问题，只要训练block5(输出层是必须训练的)就可以得到不错的结果(准确率95%)，但是糖网检测只训练block5结果不理想，所以直接训练全部。当然，初始参数都是基于imageNet训练出来的参数的

# 二分类结果
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
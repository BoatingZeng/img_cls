分析过程看record.md文件

kaggle比赛地址(糖网检测)：https://www.kaggle.com/c/diabetic-retinopathy-detection

# 依赖版本
测试环境的依赖版本，其他环境未测试

* python：3.6
* TensorFlow(GPU版本)：1.7.0
* Keras：2.1.3
* numpy：1.14.2
* cv2：3.4.0
* flask：1.0.2

# predict_online的使用方法
```
python3 predict_online.py -cp predict_online_config.json
```

测试URL：

* localhost:5000/：用于上传图片的表单页面
* localhost:5000/upload：上传图片的URL(POST)

命令参数说明：

* --config_path(-cp)：配置文件路径

predict_online_config.json.example的说明：

* img_height、img_width：这个是cnn接受的图片大小，传过来的图片会被处理成这个大小。这两个值要跟训练出来的weights配对。
* weights_path_2_cls、weights_path_4_cls：训练出来的weights路径。使用了两个模型来进行预测，二分类判别是否有病，四分类判别程度。
* ~~tem_dir：临时文件目录路径，填`null`就是系统默认。中间要保存一次cv2的图片然后再读取，所以需要临时文件，预测后就会删除。~~
* gpu_fraction：初始gpu占用率，设置了allow_growth=True，如果有需要，还会涨，如果不允许上涨的话，不够用就会死掉。如果设置gpu_fraction为0，就不使用GPU。
* port：server端口

备注：
设置gpu_fraction为0，就不使用GPU，通过`os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`实现。

训练好的weights：https://pan.baidu.com/s/1y9V2NQuX-o38wh5BP063Jw 

密码：2fnx

## ~~关于为什么需要临时文件~~
~~原始的图片格式是jpeg，在训练的时候，全部图片都用cv2进行过处理(处理后得到的图片矩阵称作**O**)，并且保存到硬盘，格式依然是jpeg。然后无论用什么api读取这个处理后的图片，都无法得到原本的矩阵**O**，而是得到一个矩阵**N**(N!=O，有一定程度的偏差)。这个估计是保存成jpeg的时候，因为压缩导致的(已经排除是RGB和BGR的不同)。因为训练的时候是使用这些处理过后的jpeg，所以最后输入去训练神经网络的是矩阵**N**。所以预测的时候，也必须把矩阵**O**保存成jpeg再读取出来，暂时不知道其他办法模拟这种压缩损失。如果用**O**来预测，结果会有不同程度的偏差。经过测试，如果保存成png，那么再读取出来的矩阵还是原来的**O**。~~

另外，keras用pil处理图片。pil和cv2的resize操作，即使都是设置interpolation为nearest，结果也是不同的。

经过对比，其实jpeg的压缩对结果影响不是很大。理论上，神经网络也不应该对这种差异太敏感。所以就不再保存jpeg临时文件。
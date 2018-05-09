分析过程看record.md文件

kaggle比赛地址(糖网检测)：https://www.kaggle.com/c/diabetic-retinopathy-detection

# predict_online的使用方法
```
python3 predict_online.py -cp predict_online_config.json
```

predict_online_config.json.example的说明：

* img_height、img_width：这个是cnn接受的图片大小，传过来的图片会被处理成这个大小。这两个值要跟训练出来的weights配对。
* weights_path_2_cls、weights_path_4_cls：训练出来的weights路径。使用了两个模型来进行预测，二分类判别是否有病，四分类判别程度。
* tem_dir：临时文件路径，填`null`就是系统默认。中间要保存一次cv2的图片然后再读取，所以需要临时文件，预测后就会删除。

训练好的weights：https://pan.baidu.com/s/1y9V2NQuX-o38wh5BP063Jw 

密码：2fnx
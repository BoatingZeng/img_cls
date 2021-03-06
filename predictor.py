import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import tempfile
import os
import tensorflow as tf

from models import vgg16


def process_img(img, scale=300, output_shape=(512, 512)):

    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    a = cv2.resize(img, (0, 0), fx=s, fy=s)

    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)

    b = np.zeros(a.shape)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    a = a * b + 128 * (1 - b)

    height = a.shape[0]
    width = a.shape[1]
    if height < width:
        a = a[:, (width - height) // 2:(width + height) // 2, :]
    else:
        a = a[(height - width) // 2:(width + height) // 2, :, :]

    # 把图片保存到临时目录
    # fd, tem = tempfile.mkstemp(suffix='.jpeg', dir=tempdir)
    # cv2.imwrite(tem, a)

    # 用keras的api读取并且处理刚才的图片
    # a = load_img(tem, target_size=output_shape)

    # 不用临时文件
    a = a[..., ::-1]
    a = array_to_img(a, data_format='channels_last', scale=False).resize(output_shape)
    a = img_to_array(a, data_format='channels_last')

    # 用keras的api正则化
    datagen = ImageDataGenerator(samplewise_center=True, rescale=1. / 255)
    a = datagen.standardize(a)

    # 变成(batch_size, height, width, channel)，batch_size=1
    a = np.expand_dims(a, axis=0)

    # 记得删除临时图片
    # os.close(fd)
    # os.remove(tem)
    return a


# p = Predictor(config)
# img = cv2.imread('E:/work/data/Diabetic_Retinopathy_Detection/sample/all/13_left.jpeg')
# p.do_predict(img) 返回0、1、2、3、4
class Predictor:
    def __init__(self, config):
        weights_path_2_cls = config['weights_path_2_cls']
        weights_path_4_cls = config['weights_path_4_cls']

        self.img_height = config['img_height']
        self.img_width = config['img_width']

        self.graph = tf.get_default_graph()

        self.model_2_cls = vgg16(input_shape=(self.img_height, self.img_width, 3), class_num=2, weights_path=weights_path_2_cls)
        self.model_4_cls = vgg16(input_shape=(self.img_height, self.img_width, 3), class_num=4, weights_path=weights_path_4_cls)

    def do_predict(self, img, threshold_2_cls=0.5):
        img = process_img(img, output_shape=(self.img_height, self.img_width))
        with self.graph.as_default():
            result_2_cls = self.model_2_cls.predict(img, batch_size=1)[0]

        healthy_rate = result_2_cls[0]
        ill_rate = result_2_cls[1]

        if ill_rate > threshold_2_cls:
            result_2_cls = 1
        else:
            result_2_cls = 0

        if result_2_cls == 1:
            with self.graph.as_default():
                result_4_cls = self.model_4_cls.predict(img, batch_size=1)[0]

            result_4_cls = int(result_4_cls.argmax()+1)
            result = result_4_cls
            rate = ill_rate
        else:
            result = result_2_cls
            rate = healthy_rate

        return result, rate

import cv2
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tempfile
import json
from wsgiref.simple_server import make_server
import cgi

from models import vgg16


def process_img(img, scale=300, output_shape=(512, 512), tempdir='/temp'):

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
    a = a[:, (width - height) // 2:(width + height) // 2, :]

    # 把图片保存到临时目录
    tem = tempfile.mktemp(suffix='.jpeg', dir=tempdir)
    cv2.imwrite(tem, a)

    # 用keras的api读取并且处理刚才的图片
    a = load_img(tem, target_size=output_shape)
    datagen = ImageDataGenerator(samplewise_center=True, rescale=1. / 255)
    a = img_to_array(a, data_format='channels_last')
    a = datagen.standardize(a)

    # 变成(batch_size, height, width, channel)，batch_size=1
    a = np.expand_dims(a, axis=0)
    # 记得删除临时图片
    os.remove(tem)
    return a


# p = Predictor('predict_online_config.json')
# img = cv2.imread('E:/work/data/Diabetic_Retinopathy_Detection/sample/all/13_left.jpeg')
# p.do_predict(img) 返回0、1、2、3、4
class Predictor:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        weights_path_2_cls = config['weights_path_2_cls']
        weights_path_4_cls = config['weights_path_4_cls']

        self.img_height = config['img_height']
        self.img_width = config['img_width']

        self.tem_dir = config['tem_dir']
        self.model_2_cls = vgg16(input_shape=(self.img_height, self.img_width, 3), class_num=2, weights_path=weights_path_2_cls)
        self.model_4_cls = vgg16(input_shape=(self.img_height, self.img_width, 3), class_num=4, weights_path=weights_path_4_cls)

    def do_predict(self, img):
        img = process_img(img, output_shape=(self.img_height, self.img_width), tempdir=self.tem_dir)
        result_2_cls = self.model_2_cls.predict(img, batch_size=1)[0]
        result_4_cls = self.model_4_cls.predict(img, batch_size=1)[0]

        # 二分类就用threshold=0.5
        result_2_cls = int(result_2_cls.argmax())
        result_4_cls = int(result_4_cls.argmax()+1)

        if result_2_cls == 0:
            result = result_2_cls
        else:
            result = result_4_cls

        return result


p = Predictor('predict_online_config.json')


def application(environ, start_response):
    if environ['REQUEST_METHOD'] == 'POST':
        # 保存upload上来的文件
        fields = cgi.FieldStorage(fp=environ['wsgi.input'], environ=environ, keep_blank_values=1)
        fileitem = fields['file']
        fn = os.path.basename(fileitem.filename)
        up_file = os.path.join(p.tem_dir, fn)
        f = open(up_file, 'wb')
        f.write(fileitem.file.read())
        f.close()

        # 读取刚才upload的文件并预测
        img = cv2.imread(up_file)
        result = p.do_predict(img)

        # 删除upload的文件
        os.remove(up_file)

        if result == 0:
            color = 1
        else:
            color = 2
        data = {
            "data": [
                {
                    "status": color,
                    "text": str(result),
                    "label": "result"
                },
                {
                    "status": color,
                    "text": "60%",
                    "label": "rate"
                }
            ]
        }
        content_type = 'application/json'
        body = json.dumps(data)
    else:
        content_type = 'text/html;charset=UTF-8'
        body = '<form id="upload" name="upload" method="POST" enctype="multipart/form-data">' \
               '<input type="file" name="file" />' \
               '<input type="submit" value="提交">' \
               '</form>'

    status = '200 OK'
    start_response(status, [('Content-Type', content_type)])

    return [body.encode('utf-8')]


httpd = make_server('', 8000, application)
print('Serving HTTP on port 8000...')
# 开始监听HTTP请求:
httpd.serve_forever()

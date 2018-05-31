import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import tensorflow as tf
import argparse
import json
from flask import Flask, jsonify, request


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


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)
        return graph


class PredictorTF:
    def __init__(self, config):
        gpu_fraction = config['gpu_fraction']

        if gpu_fraction == 0:
            tf_conf = tf.ConfigProto(
                device_count={'GPU': 0}
            )
        else:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                        allow_growth=True)
            tf_conf = tf.ConfigProto(gpu_options=gpu_options)

        weights_path_2_cls = config['weights_path_2_cls']
        weights_path_4_cls = config['weights_path_4_cls']

        self.img_height = config['img_height']
        self.img_width = config['img_width']

        graph_cls2 = load_graph(weights_path_2_cls)
        self.sess_cls2 = tf.Session(graph=graph_cls2, config=tf_conf)
        self.input_cls2 = graph_cls2.get_tensor_by_name('import/input_1:0')
        self.output_cls2 = graph_cls2.get_tensor_by_name('import/output_predictions_cls2/Softmax:0')

        graph_cls4 = load_graph(weights_path_4_cls)
        self.sess_cls4 = tf.Session(graph=graph_cls4, config=tf_conf)
        self.input_cls4 = graph_cls4.get_tensor_by_name('import/input_1:0')
        self.output_cls4 = graph_cls4.get_tensor_by_name('import/output_predictions_cls4/Softmax:0')

    def do_predict(self, img, threshold_2_cls=0.5):
        img = process_img(img, output_shape=(self.img_height, self.img_width))

        result_2_cls = self.sess_cls2.run(self.output_cls2, feed_dict={self.input_cls2: img})[0]

        healthy_rate = result_2_cls[0]
        ill_rate = result_2_cls[1]

        if ill_rate > threshold_2_cls:
            result_2_cls = 1
        else:
            result_2_cls = 0

        if result_2_cls == 1:
            result_4_cls = self.sess_cls4.run(self.output_cls4, feed_dict={self.input_cls4: img})
            result_4_cls = int(result_4_cls.argmax() + 1)
            result = result_4_cls
            rate = ill_rate
        else:
            result = result_2_cls
            rate = healthy_rate

        return result, rate


# 读取配置文件
parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--config_path', type=str, default='predict_online_tf_config.json')
args = parser.parse_args()

with open(args.config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

p = PredictorTF(config)

# Flask服务器部分
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return '''<form action="/upload" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" />
            <input type="submit" value="提交" />
            </form>'''


@app.route('/upload', methods=['POST'])
def upload():
    # 直接从流里读取图片
    stream = request.files['file'].read()
    img = cv2.imdecode(np.fromstring(stream, np.uint8), cv2.IMREAD_COLOR)
    result, rate = p.do_predict(img, config['threshold_2_cls'])

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
                "text": "{0}%".format(int(rate*100)),
                "label": "rate"
            }
        ]
    }
    return jsonify(data)


app.run(port=config['port'], host='0.0.0.0')

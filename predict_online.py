import cv2
import numpy as np
from flask import Flask, jsonify
from flask import request
import argparse
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


from predictor import Predictor

p = None
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
    result, rate = p.do_predict(img)

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


def get_session(gpu_fraction=0.5):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='predict_online_config.json')
    parser.add_argument('-gf', '--gpu_fraction', type=float, default=0.5)
    args = parser.parse_args()

    ktf.set_session(get_session(args.gpu_fraction))

    p = Predictor(args.config_path)
    app.run()

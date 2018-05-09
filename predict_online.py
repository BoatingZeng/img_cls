import cv2
import os
import json
import io
import numpy as np
from flask import Flask, jsonify
from flask import request
import argparse

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cp', '--config_path', type=str, default='predict_online_config.json')
    args = parser.parse_args()
    p = Predictor(args.config_path)
    app.run()

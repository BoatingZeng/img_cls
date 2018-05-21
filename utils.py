import shutil
import os
import random
import pandas as pd
import cv2, glob
import numpy as np
import tensorflow as tf
import json
from models import vgg16


# import and use in python console
# split_data('E:/work/data/driver_detection/train', 'E:/work/data/driver_detection/validation', 0.2)
# this function do shuffle during splitting
def split_data(src, des, rate):
    if len(os.listdir(des)) != 0:
        print('{0} is not empty'.format(des))
        return

    classes = os.listdir(src)
    print('classes num: {0}'.format(len(classes)))
    print(classes)

    for cls in classes:
        src_cls_path = os.path.join(src, cls)
        cls_imgs = os.listdir(src_cls_path)
        src_count = len(cls_imgs)
        des_count = int(src_count*rate)
        print('----------------------------------')
        print('{0} img num: {1}'.format(cls, src_count))
        des_cls_path = os.path.join(des, cls)

        os.mkdir(des_cls_path)

        move_count = 0
        progress = list(range(10, 0, -1))
        for img in cls_imgs:
            ra = random.random()
            if ra <= rate:
                src_img_path = os.path.join(src_cls_path, img)
                des_img_path = os.path.join(des_cls_path, img)
                shutil.move(src_img_path, des_img_path)
                move_count += 1
                if move_count == des_count:
                    print('{0} finish'.format(cls))
                    break
                if progress[len(progress)-1] < move_count/des_count*10:
                    print('progress: {0}%'.format(move_count/des_count*100))
                    progress.pop()
        print('real move num: {0}'.format(move_count))
        print('need to move: {0}'.format(des_count))


# 这个函数用来把糖网图片按照trainLables.csv的标注分放到不同目录
# split_dia_data('E:/work/data/Diabetic_Retinopathy_Detection/sample',
#                'E:/work/data/Diabetic_Retinopathy_Detection/all',
#                'E:/work/data/Diabetic_Retinopathy_Detection/trainLabels.csv')
def split_dia_data(src, des, lable_path):
    # 创建分类目录
    classes = ['0', '1', '2', '3', '4']
    for cls in classes:
        os.mkdir(os.path.join(des, cls))

    lables = pd.read_csv(lable_path)
    miss_count = 0
    for row in lables.values:
        img_name = row[0]+'.jpeg'
        cls = str(row[1])
        src_img_path = os.path.join(src, img_name)
        if os.path.exists(src_img_path):
            des_img_path = os.path.join(des, cls, img_name)
            shutil.move(src_img_path, des_img_path)
        else:
            miss_count += 1
            print(src_img_path)
    if miss_count > 0:
        print('以上文件不存在，总数：{0}'.format(miss_count))


def scaleRadius(img, scale):
    x = img[img.shape[0]//2, :, :].sum(1)
    r = (x > x.mean()/10).sum()/2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


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
                if height < width:
                    a = a[:, (width - height) // 2:(width + height) // 2, :]
                else:
                    a = a[(height - width) // 2:(width + height) // 2, :, :]
                basename = os.path.basename(f)
                newpath = os.path.join(new_class_dir, basename)
                cv2.imwrite(newpath, a)
            except:
                print(f)


# 把model保存成protocol buffer格式(包含参数)
def save_freeze_model(model_config_path, name):
    with open(model_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with tf.Session() as sess:
        if config['model_type'] == 'vgg16':
            model = vgg16(input_shape=(config['img_height'], config['img_width'], 3), class_num=config['class_num'], weights_path=config['weights_path'])
        else:
            raise ValueError('model_type error!')

        graph_def = sess.graph.as_graph_def()
        graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [node.op.name for node in model.outputs])
        tf.train.write_graph(graph_or_graph_def=graph_def, logdir='.', name=name, as_text=False)

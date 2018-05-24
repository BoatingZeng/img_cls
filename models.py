from keras.applications import ResNet50, VGG16, Xception
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model

import tensorflow as tf
import json


def vgg16(input_shape=(224, 224, 3), class_num=2, weights_path=None):
    if weights_path is None:
        weights = 'imagenet'
    else:
        weights = None
    base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)

    x = Flatten(name='output_flatten')(base_model.output)
    x = Dense(256, activation='relu', name='output_fc_cls'+str(class_num))(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='softmax', name='output_predictions_cls'+str(class_num))(x)

    model = Model(inputs=base_model.input, outputs=x, name='vgg16_cls'+str(class_num))
    if weights_path is not None:
        print('load weights from: '+weights_path)
        model.load_weights(weights_path)

    return model


def resnet50(input_shape=(224, 224, 3), class_num=2, weights_path=None):
    if weights_path is None:
        weights = 'imagenet'
    else:
        weights = None
    base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape)

    x = Flatten(name='output_flatten')(base_model.output)
    x = Dense(class_num, activation='softmax', name='output_predictions_cls'+str(class_num))(x)

    model = Model(inputs=base_model.input, outputs=x, name='resnet50_cls'+str(class_num))
    if weights_path is not None:
        print('load weights from: ' + weights_path)
        model.load_weights(weights_path)

    return model


def vgg16_reg(input_shape=(224, 224, 3), weights_path=None):
    if weights_path is None:
        weights = 'imagenet'
    else:
        weights = None
    base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)

    x = Flatten(name='output_flatten')(base_model.output)
    x = Dense(256, activation='relu', name='output_fc_reg')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name='output_predictions_reg')(x)

    model = Model(inputs=base_model.input, outputs=x, name='vgg16_reg')
    if weights_path is not None:
        print('load weights from: '+weights_path)
        model.load_weights(weights_path)

    return model


def xception(input_shape=(224, 224, 3), class_num=2, weights_path=None):
    if weights_path is None:
        weights = 'imagenet'
    else:
        weights = None
    base_model = Xception(weights=weights, include_top=False, input_shape=input_shape)

    x = GlobalAveragePooling2D(name='avg_pool')(base_model.output)
    x = Dense(class_num, activation='softmax', name='output_predictions_cls'+str(class_num))(x)

    model = Model(inputs=base_model.input, outputs=x, name='vgg16_cls'+str(class_num))
    if weights_path is not None:
        print('load weights from: '+weights_path)
        model.load_weights(weights_path)

    return model


def vgg16_large(input_shape=(224, 224, 3), class_num=2, weights_path=None):
    if weights_path is None:
        weights = 'imagenet'
    else:
        weights = None
    base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)

    x = Flatten(name='output_flatten')(base_model.output)
    x = Dense(512, activation='relu', name='output_fc_1_cls'+str(class_num))(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='output_fc_2_cls' + str(class_num))(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='softmax', name='output_predictions_cls'+str(class_num))(x)

    model = Model(inputs=base_model.input, outputs=x, name='vgg16_cls'+str(class_num))
    if weights_path is not None:
        print('load weights from: '+weights_path)
        model.load_weights(weights_path)

    return model


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

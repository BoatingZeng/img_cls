from keras.applications import ResNet50, VGG16, Xception
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, LeakyReLU
from keras.models import Model
from keras_contrib.applications.resnet import ResNet

import tensorflow as tf
import json
import os


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
    base_model = ResNet50(weights=weights, include_top=False, input_shape=input_shape, pooling='avg')

    # x = Flatten(name='output_flatten')(base_model.output)
    x = Dense(class_num, activation='softmax', kernel_initializer="he_normal", name='output_predictions_cls'+str(class_num))(base_model.output)

    model = Model(inputs=base_model.input, outputs=x, name='resnet50_cls'+str(class_num))
    if weights_path is not None:
        print('load weights from: ' + weights_path)
        model.load_weights(weights_path)

    return model


def resnet18(input_shape=(224, 224, 3), class_num=2, weights_path=None):
    base_model = ResNet(input_shape, class_num, 'basic', repetitions=[2, 2, 2, 2], include_top=False)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(units=class_num, activation='softmax', kernel_initializer="he_normal", name='output_predictions_cls'+str(class_num))(x)

    model = Model(inputs=base_model.input, outputs=x, name='resnet18_cls' + str(class_num))

    if weights_path is not None:
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


def vgg16_LeakyReLU(input_shape=(224, 224, 3), class_num=2, leaky_alpha=0.5, weights_path=None):
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    base_model = Model(img_input, x, name='vgg16_leaky')

    if weights_path is None:
        base_weights_path = os.path.join(os.path.expanduser('~'), '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
        base_model.load_weights(base_weights_path)

    x = Flatten(name='output_flatten')(base_model.output)
    x = Dense(256, name='output_fc_cls' + str(class_num))(x)
    x = LeakyReLU(alpha=leaky_alpha)(x)
    x = Dropout(0.5)(x)
    x = Dense(class_num, activation='softmax', name='output_predictions_cls' + str(class_num))(x)

    model = Model(inputs=base_model.input, outputs=x, name='vgg16_leaky_cls' + str(class_num))
    if weights_path is not None:
        print('load weights from: '+weights_path)
        model.load_weights(weights_path)

    return model


# 把model保存成protocol buffer格式(包含参数)
def save_freeze_model(model_config_path, name):
    with open(model_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with tf.Session() as sess:
        model_type = config['model_type']
        if model_type == 'vgg16':
            model = vgg16(input_shape=(config['img_height'], config['img_width'], 3),
                          class_num=config['class_num'], weights_path=config['weights_path'])

        elif model_type == 'resnet50':
            model = resnet50(input_shape=(config['img_height'], config['img_width'], 3),
                             class_num=config['class_num'], weights_path=config['weights_path'])

        elif model_type == 'resnet18':
            model = resnet18(input_shape=(config['img_height'], config['img_width'], 3),
                             class_num=config['class_num'], weights_path=config['weights_path'])

        else:
            raise ValueError('model_type error!')

        graph_def = sess.graph.as_graph_def()
        graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, [node.op.name for node in model.outputs])
        tf.train.write_graph(graph_or_graph_def=graph_def, logdir='.', name=name, as_text=False)

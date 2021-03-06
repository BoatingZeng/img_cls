from keras.preprocessing.image import ImageDataGenerator
import json
import argparse
import os
import pandas as pd

from models import vgg16, resnet50, vgg16_LeakyReLU, resnet18


def predict_driver(model, train_config, predict_data_dir, num_predict_samples, batch_size, result_path):
    img_height = train_config['img_height']
    img_width = train_config['img_width']

    predict_datagen = ImageDataGenerator(samplewise_center=True, rescale=1. / 255)

    predict_generator = predict_datagen.flow_from_directory(
        predict_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    results = model.predict_generator(predict_generator, steps=num_predict_samples // batch_size, verbose=1)

    columns = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    re_frame = pd.DataFrame(results, columns=columns)

    filenames = predict_generator.filenames
    img_names = []
    for i in range(num_predict_samples):
        path_name = filenames[i]
        name = os.path.basename(path_name)
        img_names.append(name)

    re_frame['img'] = img_names
    header = ['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    re_frame = re_frame[header]
    re_frame.to_csv(result_path, index=False)


def predict_diabetic(model, train_config, predict_data_dir, num_predict_samples, batch_size, result_path):
    if len(os.listdir(predict_data_dir)) == train_config['class_num']:
        has_true_class = True
    else:
        has_true_class = False

    img_height = train_config['img_height']
    img_width = train_config['img_width']

    predict_datagen = ImageDataGenerator(samplewise_center=True, rescale=1. / 255)

    predict_generator = predict_datagen.flow_from_directory(
        predict_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    steps = num_predict_samples // batch_size
    results = model.predict_generator(predict_generator, steps=steps, verbose=1)
    results = results.argmax(axis=1)

    columns = ['level']
    re_frame = pd.DataFrame(results, columns=columns)

    filenames = predict_generator.filenames
    img_names = []
    true_cls_list = []
    for i in range(steps * batch_size):
        path_name = filenames[i]
        name = os.path.basename(path_name)
        if has_true_class:
            true_class = int(os.path.basename(os.path.dirname(path_name)))
            true_cls_list.append(true_class)
        # 不要后缀
        name, _ = os.path.splitext(name)
        img_names.append(name)

    re_frame['image'] = img_names
    header = ['image', 'level']
    if has_true_class:
        re_frame['true_class'] = true_cls_list
        header.append('true_class')

    re_frame = re_frame[header]
    re_frame.to_csv(result_path, index=False)


def predict_diabetic_prob(model, train_config, predict_data_dir, num_predict_samples, batch_size, result_path):
    class_num = train_config['class_num']
    if len(os.listdir(predict_data_dir)) == class_num:
        has_true_class = True
    else:
        has_true_class = False

    img_height = train_config['img_height']
    img_width = train_config['img_width']

    if class_num == 2:
        columns = ['not_ill', 'ill']
    elif class_num == 4:
        columns = ['1', '2', '3', '4']
    else:
        raise ValueError('class_num error')

    header = ['image']
    header.extend(columns)

    predict_datagen = ImageDataGenerator(samplewise_center=True, rescale=1. / 255)

    predict_generator = predict_datagen.flow_from_directory(
        predict_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    steps = num_predict_samples // batch_size
    results = model.predict_generator(predict_generator, steps=steps, verbose=1)

    re_frame = pd.DataFrame(results, columns=columns)

    filenames = predict_generator.filenames
    img_names = []
    true_cls_list = []
    for i in range(steps * batch_size):
        path_name = filenames[i]
        name = os.path.basename(path_name)
        if has_true_class:
            true_class = int(os.path.basename(os.path.dirname(path_name)))
            true_cls_list.append(true_class)
        # 不要后缀
        name, _ = os.path.splitext(name)
        img_names.append(name)

    re_frame['image'] = img_names

    if has_true_class:
        re_frame['true_class'] = true_cls_list
        header.append('true_class')

    re_frame = re_frame[header]
    re_frame.to_csv(result_path, index=False)


parser = argparse.ArgumentParser()
parser.add_argument('-tcp', '--train_config_path', type=str, default='train_config.json')
parser.add_argument('-pdd', '--predict_data_dir', type=str, default=None)
parser.add_argument('-nps', '--num_predict_samples', type=int, default=64)
parser.add_argument('-bs', '--batch_size', type=int, default=2)
parser.add_argument('--task', type=str, default='diabetic_prob')
parser.add_argument('-rp', '--result_path', type=str, default='result.csv')

args = parser.parse_args()

with open(args.train_config_path, 'r', encoding='utf-8') as f:
    train_config = json.load(f)

if os.path.exists(train_config['weights_path']):
    weights_path = train_config['weights_path']
else:
    raise ValueError('weights_path not exit!')

model_type = train_config['model_type']
if model_type == 'vgg16':
    model = vgg16(input_shape=(train_config['img_height'], train_config['img_width'], 3), class_num=train_config['class_num'], weights_path=weights_path)
elif model_type == 'resnet50':
    model = resnet50(input_shape=(train_config['img_height'], train_config['img_width'], 3), class_num=train_config['class_num'], weights_path=weights_path)
elif model_type == 'resnet18':
    model = resnet18(input_shape=(train_config['img_height'], train_config['img_width'], 3), class_num=train_config['class_num'], weights_path=weights_path)
elif model_type == 'vgg16_LeakyReLU':
    model = vgg16_LeakyReLU(input_shape=(train_config['img_height'], train_config['img_width'], 3),
                            class_num=train_config['class_num'], leaky_alpha=train_config['leaky_alpha'],
                            weights_path=weights_path)
else:
    raise ValueError('model_type error!')

result_path = args.result_path
if args.task == 'driver':
    predict_driver(model, train_config, args.predict_data_dir, args.num_predict_samples, args.batch_size, result_path)
elif args.task == 'diabetic':
    predict_diabetic(model, train_config, args.predict_data_dir, args.num_predict_samples, args.batch_size, result_path)
elif args.task == 'diabetic_prob':
    predict_diabetic_prob(model, train_config, args.predict_data_dir, args.num_predict_samples, args.batch_size, result_path)
else:
    raise ValueError('check task name')

from keras.preprocessing.image import ImageDataGenerator
import json
import argparse
import os
import pandas as pd

from models import vgg16


def predict(model, train_config, predict_data_dir, num_predict_samples, batch_size):
    img_height = train_config['img_height']
    img_width = train_config['img_width']

    predict_datagen = ImageDataGenerator(rescale=1. / 255)

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
        name = os.path.split(path_name)[-1]
        img_names.append(name)

    re_frame['img'] = img_names
    header = ['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    re_frame = re_frame[header]
    re_frame.to_csv('results.csv', index=False)


parser = argparse.ArgumentParser()
parser.add_argument('-tcp', '--train_config_path', type=str, default='train_config.json')
parser.add_argument('-pdd', '--predict_data_dir', type=str, default=None)
parser.add_argument('-nps', '--num_predict_samples', type=int, default=64)
parser.add_argument('-bs', '--batch_size', type=int, default=2)

args = parser.parse_args()

with open(args.train_config_path, 'r', encoding='utf-8') as f:
    train_config = json.load(f)

model_type = train_config['model_type']
if model_type == 'vgg16':
    model = vgg16(classes=train_config['class_num'], weights_path=train_config['weights_path'])
else:
    raise ValueError('model_type error!')

predict(model, train_config, args.predict_data_dir, args.num_predict_samples, args.batch_size)

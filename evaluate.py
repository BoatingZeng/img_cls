from keras.preprocessing.image import ImageDataGenerator
import json
import argparse
import os

from models import vgg16


def evaluate(model, train_config, test_data_dir, evaluate_img_num, batch_size):
    weights_path = train_config['weights_path']
    img_height = train_config['img_height']
    img_width = train_config['img_width']

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # load best weights before evaluate
    print('loading best weights to evaluate')
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print('evaluate folder: '+test_data_dir)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')
    score = model.evaluate_generator(test_generator, steps=evaluate_img_num // batch_size)
    print('score')
    print(score)
    test_score_path = weights_path + '.score.txt'
    with open(test_score_path, 'w', encoding='utf-8') as f:
        f.write(str(score))


parser = argparse.ArgumentParser()
parser.add_argument('-tcp', '--train_config_path', type=str, default='train_config.json')
parser.add_argument('-tdd', '--test_data_dir', type=str, default=None)
parser.add_argument('-ein', '--evaluate_img_num', type=int, default=4432)
parser.add_argument('-bs', '--batch_size', type=int, default=8)

args = parser.parse_args()

with open(args.train_config_path, 'r', encoding='utf-8') as f:
    train_config = json.load(f)

model_type = train_config['model_type']

if os.path.exists(train_config['weights_path']):
    weights_path = train_config['weights_path']
else:
    raise ValueError('weights_path not exit!')

if model_type == 'vgg16':
    model = vgg16(class_num=train_config['class_num'], weights_path=weights_path)
else:
    raise ValueError('model_type error!')

evaluate(model, train_config, args.test_data_dir, args.evaluate_img_num, args.batch_size)

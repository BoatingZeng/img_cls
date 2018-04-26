from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import json
from keras.callbacks import ModelCheckpoint
import argparse
import os

from models import vgg16, resnet50


def train(model, train_config):
    lr = train_config['lr']
    momentum = train_config['momentum']
    train_data_dir = train_config['train_data_dir']
    validation_data_dir = train_config['validation_data_dir']
    img_height = train_config['img_height']
    img_width = train_config['img_width']
    batch_size = train_config['batch_size']
    num_train_samples = train_config['num_train_samples']
    num_validation_samples = train_config['num_validation_samples']
    epochs = train_config['epochs']
    weights_path = train_config['weights_path']
    class_weight = train_config['class_weight']

    sgd = SGD(lr=lr, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rotation_range=360,  # 用于眼球图片
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    checkpointer = ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min')

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=num_validation_samples // batch_size,
        callbacks=[checkpointer],
        class_weight=class_weight)


parser = argparse.ArgumentParser()
parser.add_argument('-tcp', '--train_config_path', type=str, default='train_config.json')

args = parser.parse_args()

with open(args.train_config_path, 'r', encoding='utf-8') as f:
    train_config = json.load(f)

model_type = train_config['model_type']

if os.path.exists(train_config['weights_path']):
    weights_path = train_config['weights_path']
else:
    weights_path = None

if model_type == 'vgg16':
    model = vgg16(input_shape=(train_config['img_height'], train_config['img_width'], 3), class_num=train_config['class_num'], weights_path=weights_path)
    # 冻结不训练的层
    for layer in model.layers:
        if layer.name.find('output') == 0:
            # 不冻结输出层
            continue
        if not train_config['just_train_output'] and layer.name.find('block5') == 0:
            # 不冻结block5
            continue

        layer.trainable = False

elif model_type == 'resnet50':
    model = resnet50(input_shape=(train_config['img_height'], train_config['img_width'], 3), class_num=train_config['class_num'], weights_path=weights_path)
    for layer in model.layers:
        if layer.name.find('output') == 0:
            # 不冻结输出层
            continue
        if not train_config['just_train_output'] and (layer.name.find('res5') == 0 or layer.name.find('bn') == 0):
            # stage 5 的层不冻结，batch normalization 层全部不冻结
            continue

        layer.trainable = False

else:
    raise ValueError('model_type error!')

train(model, train_config=train_config)

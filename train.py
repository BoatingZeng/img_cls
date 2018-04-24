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

    sgd = SGD(lr=lr, momentum=momentum)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
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
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max')

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
parser.add_argument('-fl', '--freeze_layer', type=int, default=15)

args = parser.parse_args()

with open(args.train_config_path, 'r', encoding='utf-8') as f:
    train_config = json.load(f)

model_type = train_config['model_type']

if os.path.exists(train_config['weights_path']):
    weights_path = train_config['weights_path']
else:
    weights_path = None

if model_type == 'vgg16':
    model = vgg16(class_num=train_config['class_num'], weights_path=weights_path)
    # 冻结不训练的层
    # vgg16各个block的分隔index: [4, 7, 11, 15, 19]
    for layer in model.layers[:args.freeze_layer]:
        layer.trainable = False
elif model_type == 'resnet50':
    model = resnet50(class_num=train_config['class_num'], weights_path=weights_path)
    # 174
    for layer in model.layers[:174]:
        layer.trainable = False
else:
    raise ValueError('model_type error!')

train(model, train_config=train_config)

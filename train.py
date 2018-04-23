from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import json
from keras.callbacks import ModelCheckpoint
import argparse

from models import vgg16


def train(model, train_config_path='train_config.json', test_data_dir=None, evaluate_img_num=2000):
    with open(train_config_path, 'r', encoding='utf-8') as f:
        train_config = json.load(f)

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
        callbacks=[checkpointer])

    if test_data_dir is not None:
        # load best weights before evaluate
        print('loading best weights to evaluate')
        model.load_weights(weights_path)
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
parser.add_argument('-cls', '--classes', type=int, default=2)
parser.add_argument('-pwp', '--pre_weights_path', type=str, default=None)
parser.add_argument('-tdd', '--test_data_dir', type=str, default=None)
parser.add_argument('-fl', '--freeze_layer', type=int, default=15)
parser.add_argument('-ein', '--evaluate_img_num', type=int, default=2000)

args = parser.parse_args()

model = vgg16(classes=args.classes, weights_path=args.pre_weights_path)

# 冻结不训练的层
# vgg16各个block的分隔index: [4, 7, 11, 15, 19]
for layer in model.layers[:args.freeze_layer]:
    layer.trainable = False
train(model, train_config_path=args.train_config_path, test_data_dir=args.test_data_dir, evaluate_img_num=args.evaluate_img_num)

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import json
from models import vgg16


def train(model, train_config_path='train_config.json', test_data_dir=None, test_score_path=None):
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

    model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=num_validation_samples // batch_size)

    model.save_weights(weights_path)

    if test_data_dir is not None:
        print('evaluate folder: '+test_data_dir)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')
        score = model.evaluate_generator(test_generator)
        print('score')
        print(score)
        if test_score_path is not None:
            with open(test_score_path, 'w', encoding='utf-8') as f:
                f.write(str(score))


model = vgg16(weights_path='weights/vgg16_cls2.h5')

# 冻结不训练的层
for layer in model.layers[:19]:
    layer.trainable = False
train(model, test_data_dir='train', test_score_path='weights/vgg16_cls2.score.txt')

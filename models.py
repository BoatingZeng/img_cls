from keras.applications import ResNet50, VGG16, VGG19
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model


def vgg16(input_shape=(224, 224, 3), classes=2, weights_path=None):
    if weights_path is None:
        weights = 'imagenet'
    else:
        weights = None
    base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)

    # 试试用和原来一样的top层
    x = Flatten(name='flatten')(base_model.output)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax', name='predictions_cls'+str(classes))(x)

    model = Model(inputs=base_model.input, outputs=x)
    if weights_path is not None:
        print('load weights from: '+weights_path)
        model.load_weights(weights_path)

    return model

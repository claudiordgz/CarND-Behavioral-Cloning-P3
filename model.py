from data_processing import get_images, adjust_properties_per_transform, adjust_angle_per_camera
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import json
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


def get_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(90, 320, 3)))
    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid',
                            input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, 3, 3,
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten(input_shape=(32, 32, 3)))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    return model


def get_images_generator(images, image_getter, BATCH_SIZE):
    """ The generator of data
        returns: Tuple(
            numpy array of BATCH_SIZE with images in it,
            numpy array of steering angles
            )
    """
    # Numpy arrays as large as expected
    IMAGE_WIDTH = 320
    IMAGE_HEIGHT = 160
    CHANNELS = 3
    batch_images = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    batch_features = np.zeros(BATCH_SIZE)
    for i, image_index in enumerate(images):
        image, features, transform, camera = image_getter(image_index)
        features, camera = adjust_properties_per_transform(features, camera, transform)
        payload = transform(image)
        features = adjust_angle_per_camera(features, camera)
        batch_position = i % BATCH_SIZE
        batch_images[batch_position] = payload
        batch_features[batch_position] = features[0]
        if batch_position+1 == BATCH_SIZE:
            yield batch_images, batch_features


def get_batch_size(total_images):
    """ We want our set of images to be divisible by the
        epochs
    """
    epochs = 16
    for i in range(16, 512):
        if total_images % i == 0:
            epochs = i
    return epochs


def samples(epochs, total):
    """ We want the samples to be a multiple of the EPOCHS
    """
    samples, i = epochs, 10
    while samples < 50000:
        samples = epochs * i
        i += 2
    return samples


def main():
    image_index_db, image_getter, batch_size = get_images('./data')
    x_train, x_test = train_test_split(image_index_db, test_size=0.4)
    x_test, x_evaluate = train_test_split(x_test, test_size=0.4)
    x_train = shuffle(x_train)
    model = get_model()
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    # MAGIC NUMBERS
    BATCH_SIZE = get_batch_size(len(x_train))
    SAMPLES_PER_EPOCH = samples(BATCH_SIZE, len(x_train))
    VALIDATION_BATCH_SIZE = get_batch_size(len(x_test))
    VALIDATION_SAMPLES_PER_EPOCH = samples(VALIDATION_BATCH_SIZE, len(x_test))
    EVALUATION_BATCH = get_batch_size(len(x_evaluate))
    EVALUATION_SAMPLES = samples(EVALUATION_BATCH, len(x_evaluate))
    EPOCHS = 10
    names = ['IMAGES IN TRAINING', 'IMAGES IN VALIDATION SET',
             'IMAGES IN EVALUATION SET',
             'BATCH SIZE', 'SAMPLES PER EPOCH', 'VALIDATION BATCH SIZE',
             'VALIDATION SAMPLES PER EPOCH', 'EVALUATION BATCH SIZE',
             'EVALUATION SAMPLES PER EPOCH']
    magic_numbers = [len(x_train), len(x_test), len(x_evaluate),
                     BATCH_SIZE, SAMPLES_PER_EPOCH, VALIDATION_BATCH_SIZE,
                     VALIDATION_SAMPLES_PER_EPOCH, EVALUATION_BATCH,
                     EVALUATION_SAMPLES]
    for name, value in zip(names, magic_numbers):
        print("{0:<30s} {1}".format(name, value))
    training_generator = get_images_generator(x_train, image_getter, BATCH_SIZE)
    validation_generator = get_images_generator(x_test, image_getter, VALIDATION_BATCH_SIZE)
    evaluation_generator = get_images_generator(x_evaluate, image_getter, EVALUATION_BATCH)
    history = model.fit_generator(training_generator,
                                  samples_per_epoch=SAMPLES_PER_EPOCH,
                                  verbose=2,
                                  validation_data=validation_generator,
                                  nb_val_samples=VALIDATION_SAMPLES_PER_EPOCH,
                                  nb_epoch=EPOCHS)
    print("The validation accuracy is: %.3f." % history.history['val_acc'][-1])
    metrics = model.evaluate_generator(evaluation_generator,
                                       val_samples=EVALUATION_SAMPLES)
    for metric_i in range(len(model.metrics_names)):
        metric_name = model.metrics_names[metric_i]
        metric_value = metrics[metric_i]
        print('{}: {}'.format(metric_name, metric_value))
    model.save('model.h5')
    json_string = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(json_string, f)


if __name__ == '__main__':
    main()

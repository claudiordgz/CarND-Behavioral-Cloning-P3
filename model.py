from data_processing import get_images, adjust_properties_per_transform, adjust_angle_per_camera
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os
# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf


def get_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(90, 320, 3)))
    model.add(Convolution2D(8, 3, 3,
                            border_mode='valid',
                            input_shape=(90, 320, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(8, 3, 3,
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 3, 3,
                            border_mode='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten(input_shape=(8, 8, 3)))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    return model


def get_batch_properties(images, image_getter):
    """ Order the images and the stuff we are going to
        do to them in a dictionary for easy processing
        images: a list of integers
        image_getter: a function, receives an integer, 
            returns image data and what to do to the image
    """
    r = {}
    for i in images:
        image_path, features, transform, camera = image_getter(i)
        if image_path not in r:
            r[image_path] = [(transform, features, camera)]
        else:
            r[image_path].append((transform, features, camera))
    return r


def get_images_generator(images, image_getter, BATCH_SIZE, DEBUG=False):
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
    begin_batch = 0
    N = len(images) + 1
    while begin_batch+BATCH_SIZE < N:
        i = 0
        batch_dictionary = get_batch_properties(images[begin_batch:begin_batch+BATCH_SIZE], image_getter)
        if DEBUG:
            rows = 5
            cols = BATCH_SIZE // rows
            fig, axes = plt.subplots(rows, cols)
        for img_path in batch_dictionary.keys():
            abs_path = os.path.abspath(img_path)
            img = cv2.imread(abs_path)
            for transform, features, camera in batch_dictionary[img_path]:
                features_adjusted, camera_adjusted = adjust_properties_per_transform(features, camera, transform)
                payload = transform(img)
                steer = adjust_angle_per_camera(features_adjusted, camera_adjusted)
                if DEBUG:
                    ax = axes[i // cols, i % cols]
                    ax.imshow(payload)
                    s = 'Steer %.5f' %steer[0]
                    label = 'Camera - {camera}\nTransform - {transform}\n{steer}'.format(camera=camera_adjusted, transform=transform.__name__, steer=s)
                    ax.set_title(label)
                    ax.axis('off')
                batch_position = i % BATCH_SIZE
                batch_images[batch_position] = payload
                batch_features[batch_position] = steer[0]
                i += 1    
        if DEBUG:
            plt.subplots_adjust(hspace=0.5)
            plt.show()      
        begin_batch += BATCH_SIZE
        print('new begin: ', begin_batch)
        yield batch_images, batch_features


def get_batch_size(total_images):
    """ We want our set of images to be divisible by the
        epochs
    """
    MIN_EPOCHS = 1
    MAX_SIZE = 50
    for i in range(MIN_EPOCHS, MAX_SIZE):
        if total_images % i == 0:
            MIN_EPOCHS = i
    return MIN_EPOCHS


def samples(epochs, total):
    """ We want the samples to be a multiple of the EPOCHS
    """
    samples, i = epochs, 10
    while samples < 50000:
        samples = epochs * i
        i += 2
    return samples


def main():
    image_index_db, image_getter, _ = get_images('./data')
    x_train, x_test = train_test_split(image_index_db, test_size=0.4)
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
    EPOCHS = 5
    names = ['IMAGES IN TRAINING', 'IMAGES IN VALIDATION SET',
             'BATCH SIZE', 'SAMPLES PER EPOCH', 'VALIDATION BATCH SIZE',
             'VALIDATION SAMPLES PER EPOCH']
    magic_numbers = [len(x_train), len(x_test),
                     BATCH_SIZE, SAMPLES_PER_EPOCH, VALIDATION_BATCH_SIZE,
                     VALIDATION_SAMPLES_PER_EPOCH]
    for name, value in zip(names, magic_numbers):
        print("{0:<30s} {1}".format(name, value))
    training_generator = get_images_generator(x_train, image_getter, BATCH_SIZE)
    validation_generator = get_images_generator(x_test, image_getter, VALIDATION_BATCH_SIZE)
    history = model.fit_generator(training_generator,
                                  samples_per_epoch=SAMPLES_PER_EPOCH,
                                  verbose=2,
                                  validation_data=validation_generator,
                                  nb_val_samples=VALIDATION_SAMPLES_PER_EPOCH,
                                  nb_epoch=EPOCHS)
    print("The validation accuracy is: %.3f." % history.history['val_acc'][-1])
    model.save('model.h5')
    json_string = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(json_string, f)


def test_generator():
    image_index_db, image_getter, _ = get_images('./data')
    shuffled_images = shuffle(image_index_db)
    BATCH_SIZE = 30
    sauce_generator = get_images_generator(shuffled_images, image_getter, BATCH_SIZE, True)
    for i, p in enumerate(sauce_generator):
        if i == 10:
            break


if __name__ == '__main__':
    test_generator()
    #main()


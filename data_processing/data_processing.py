import data_processing.data_preprocessing as processing
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def get_log():
    return processing.data_validation()


def read_log(p, **args):
    return processing.read_csv(p, **args)


def image_getter(df, N, N_c, N_t, D_c, D_t, features):
    """ Setups a function with the fields we need
        df: dataframe received from csv
        N: total images in dataframe
        N_c: Number of different camera pictures
        N_t: Number of transforms to be done
        D_c: Maps Int -> Camera String
        D_t: Maps Int -> Transform function
        features: Array of features to retrieve
        returns: Function to get images based on index
    """
    def getter(image_index):
        """ Curried function that receives an Int Index
            and returns properties of that image
            image_index: index of the image to use
            returns: Tuple(
                relative path to image,
                feature set of the image,
                transform function,
                camera string
            )
        """
        image_camera = image_index % N_c
        image_transformation = image_index // (N * N_c)
        image_row = image_index // (N_c*N_t)
        return df.loc[image_row, D_c[image_camera]], df.loc[image_row, features].as_matrix(), D_t[image_transformation], D_c[image_camera]
    return getter


def normalize(image_data):
    return image_data/255.-.5


def original(image_path, apply_normalize=False):
    abs_path = os.path.abspath(image_path)
    img = cv2.cvtColor(cv2.imread(abs_path), cv2.COLOR_BGR2RGB)
    return normalize(img) if apply_normalize else img


def high_brightness_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    factor = 1.25
    img[:, :, 2] = np.where(img[:, :, 2] * factor < 255, img[:, :, 2] * factor, 255)
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def high_brightness(image_path, apply_normalize=False):
    abs_path = os.path.abspath(image_path)
    img = cv2.imread(abs_path)
    img = high_brightness_image(img)
    return normalize(img) if apply_normalize else img


def low_brightness_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = img[:, :, 2] * 0.25
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def low_brightness(image_path, apply_normalize=False):
    abs_path = os.path.abspath(image_path)
    img = cv2.imread(abs_path)
    img = low_brightness_image(img)
    return normalize(img) if apply_normalize else img


def adjust_gamma_on_image(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def adjust_gamma(image_path, gamma=1.0):
    abs_path = os.path.abspath(image_path)
    img = cv2.cvtColor(cv2.imread(abs_path), cv2.COLOR_BGR2RGB)
    return adjust_gamma_on_image(img, gamma)


def __gamma_correction(image_path, correction, apply_normalize):
    if apply_normalize:
        return normalize(adjust_gamma(image_path, correction))
    else:
        return adjust_gamma(image_path, correction)


def gamma_correction_05(image_path, apply_normalize=False): return __gamma_correction(image_path, 0.5, apply_normalize)
def gamma_correction_15(image_path, apply_normalize=False): return __gamma_correction(image_path, 1.5, apply_normalize)
def gamma_correction_20(image_path, apply_normalize=False): return __gamma_correction(image_path, 2.0, apply_normalize)
def gamma_correction_25(image_path, apply_normalize=False): return __gamma_correction(image_path, 2.5, apply_normalize)


def flip_image(img):
    return np.fliplr(img)


def flip_original(image_path, apply_normalize=False):
    """ Flips the image
    """
    abs_path = os.path.abspath(image_path)
    img = cv2.cvtColor(cv2.imread(abs_path), cv2.COLOR_BGR2RGB)
    img = flip_image(img)
    return normalize(img) if apply_normalize else img


def flip_low_brightness(image_path, apply_normalize=False):
    abs_path = os.path.abspath(image_path)
    img = cv2.imread(abs_path)
    img = flip_image(img)
    img = low_brightness_image(img)
    return normalize(img) if apply_normalize else img


def flip_high_brightness(image_path, apply_normalize=False):
    abs_path = os.path.abspath(image_path)
    img = cv2.imread(abs_path)
    img = flip_image(img)
    img = high_brightness_image(img)
    return normalize(img) if apply_normalize else img


def flip_and_gamma_correction(image_path, gamma, apply_normalize):
    abs_path = os.path.abspath(image_path)
    img = cv2.cvtColor(cv2.imread(abs_path), cv2.COLOR_BGR2RGB)
    img = flip_image(img)
    img = adjust_gamma_on_image(img, gamma)
    return normalize(img) if apply_normalize else img


def flip_gamma_correction_05(image_path, apply_normalize=False): return flip_and_gamma_correction(image_path, 0.5, apply_normalize)
def flip_gamma_correction_15(image_path, apply_normalize=False): return flip_and_gamma_correction(image_path, 1.5, apply_normalize)
def flip_gamma_correction_20(image_path, apply_normalize=False): return flip_and_gamma_correction(image_path, 2.0, apply_normalize)
def flip_gamma_correction_25(image_path, apply_normalize=False): return flip_and_gamma_correction(image_path, 2.5, apply_normalize)


def adjust_angle_per_camera(features, camera):
    """ create adjusted steering measurements for the side camera images
    """
    correction = 0.13
    if camera == 'left':
        features[0] = features[-1] + correction
    elif camera == 'right':
        features[0] = features[-1] - correction
    return features


def adjust_properties_per_transform(features, camera, transform):
    """ create adjusted steering measurements for the specific transform
    """
    # Flipping - Take the opposite sign and invert left/right cameras
    if 'flip' in transform.__name__:
        features[0] = features[-1] * - 1
        if camera == 'right':
            camera = 'left'
        elif camera == 'left':
            camera = 'right'
    return features, camera


def get_images(p='./data'):
    """ Image transforms will be done on the fly. 
        To shuffle our dataset we need the full list of images.
    """
    logfile = processing.data_validation(p)
    df = read_log(logfile, skiprows=1)
    N = len(df.index)
    D_c = {
        0: 'left',
        1: 'right',
        2: 'center'
    }
    N_c = len(D_c) # left, right center cameras
    columns_w_features = processing.get_names()[N_c:]
    D_t = {
        0: original,
        1: low_brightness,
        2: high_brightness,
        3: gamma_correction_05,
        4: gamma_correction_15,
        5: gamma_correction_20,
        6: gamma_correction_25,
        7: flip_original,
        8: flip_low_brightness,
        9: flip_high_brightness,
        10: flip_gamma_correction_05,
        11: flip_gamma_correction_15,
        12: flip_gamma_correction_20,
        13: flip_gamma_correction_25
    }
    N_t = len(D_t)  # transforms we will be applying
    N_images = N*N_c*N_t
    images = list(range(0, N_images))
    get_image_data = image_getter(df, N, N_c, N_t, D_c, D_t, columns_w_features)
    return images, get_image_data, N*N_c


def test_transforms(image_index, image_getter, batch_size):
    """ We just want to check how each of the transform works, we do
        this by offsetting the number of images N * number
        of cameras N_c * transform we want to see.
        TODO: Convert following in unit tests
            - Flip should invert the steering angle -/+
            - Flip must invert cameras and modify angles accordingly
            - Images should not return normalized, that is to be handled by Keras
            - Images should not be cropped and should not specify sizes, transforms should work on any shape

    """
    for trans in range(14):
        for n_camera in range(3):
            for index in image_index[batch_size*trans+n_camera:]:
                image, features, transform, camera = image_getter(index)
                features, camera = adjust_properties_per_transform(features, camera, transform)
                print(camera, image, features, transform)
                fig, axes = plt.subplots(1, 2)
                payload_normalized = transform(image, True)
                payload = transform(image)
                print(payload.shape)
                print('features', features)
                features = adjust_angle_per_camera(features, camera)
                print('features', features)
                axes[0].imshow(payload)
                axes[1].imshow(payload_normalized)
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
                plt.show()
                break


def main():
    image_index, image_getter, batch_size = get_images('../data')
    test_transforms(image_index, image_getter, batch_size)


if __name__ == '__main__':
    main()

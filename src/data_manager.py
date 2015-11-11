__author__ = "Charlie, Josh"

from sklearn import datasets
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.misc import imfilter
from sklearn.utils import shuffle
import itertools


train_inputs1 = './data/train_inputs1.npz'
train_inputs2 = './data/train_inputs2.npz'
train_outputs = './data/train_outputs.npz'
test_inputs = './data/test_inputs.npz'


def memoize(func):
    """
    @param func: function to be decorated
    @ return: decorated function
    """
    cache = {}

    def wrapped(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapped


def available_datasets():
    """
    """
    return [
        ('iris', load_iris_data),
        ('kaggle images', load_raw_data)
    ]


def load_test_data():
    """
    :return:
    """
    X1 = np.load(train_inputs1)['train_inputs1_np']
    X2 = np.load(train_inputs2)['train_inputs2_np']
    X = np.vstack((X1, X2))
    Y = np.load(train_outputs)['train_outputs_np']
    return X, Y


@memoize
def load_iris_data():
    """
    For testing purposes. Loads a small dataset from sklearn.
    """
    iris = datasets.load_iris()
    return np.matrix(iris.data), np.array(iris.target)


def load_all_images(rotate=True):
    """
    Concatenated and shuffles images returned by load_raw_data, load_raw_mnist_images, and load_transformed_images
    :return: all available images as np.array (n x 48 x 48) where n is the number of available images
    """

    mnist_imgs, mnist_clss = load_filtered_mnist_images()
    raw_imgs, raw_clss = load_raw_data()
    imgs, clss = np.vstack(raw_imgs, mnist_imgs), np.vstack(raw_clss,  mnist_clss)
    imgs, clss = shuffle(imgs, clss)
    imgs, clss = generate_trimmed_images(imgs, clss)
    if rotate:
        imgs, clss = generate_rotated_images(imgs, clss)
    return imgs, clss


@memoize
def load_raw_data():
    """
    Delegates to load_test_data
    """
    imgs, clss = load_test_data()
    return np.reshape(imgs, (-1, 48, 48)), clss


@memoize
def load_raw_mnist_images():
    """
    :return: np.array of mnist images and np.array of mnist image classes
    The images are in raw format
    """
    imgs = np.load('./data/mnist/mnist-images.npy')
    clss = np.load('./data/mnist/mnist-classes.npy')
    return imgs, clss


def load_filtered_mnist_images():
    """
    :param *args: an option list
    :return: np.array of mnist images and np.array of mnist image classes
    The images are returned filtered by several image filters. Any filters
    that can be used with scipy.misc.imfilter may be used.
    """
    filters = ['emboss', 'blur']
    imgs, clss = load_raw_mnist_images()
    for index, img in enumerate(imgs):
        for filter in filters:
            img = imfilter(img, filter)
        imgs[index,:,:] = img
    return imgs, clss


def load_rotated_images():
    """
    :return: tuple of 3d numpy array of images and 1d numpy array of images
    """
    imgs, clss = load_raw_data()
    return generate_rotated_images(imgs, clss)


def load_transformed_images(rotated=True, trimmed=True):
    """
    :param rotated: boolean. if true, will generate rotated images
    :param trimmed: boolean. if true, will generate trimmed images
    :return: transformed images
    """
    imgs, clss = load_raw_data()
    if rotated:
        imgs, clss = generate_rotated_images(imgs, clss)
    if trimmed:
        imgs = generate_trimmed_images(imgs)
    return imgs, clss


def generate_rotated_images(images, classes):
    """
    :param images: 3d array of numpy array images (n x 48 x 48)
    :param classes: 1d array of image classifications
    :return: tuple np.array of rotated images
    """
    num_rotations = 4
    rotated_images = []
    for index in range(images.shape[0]):
        image = images[index,:,:]
        rotated_images.append(_rotations(image, num_rotations))
    return np.vstack(tuple(rotated_images)), np.repeat(classes, num_rotations)


def generate_trimmed_images(images, edge_top=2, edge_left=2, edge_bottom=2, edge_right=2):
    """
    :param images: 3d numpy array with dimension (n, 48, 48)
    :return: numpy array with dimension (n, (48-edge_left-edge_right), (48-edge_top-edge_bottom))
    where the edges of image have been removed
    """
    trimmed_images = []
    for index in range(images.shape[0]):
        image = images[index,:,:]
        trimmed_images.append(trim_edges(image, edge_top, edge_left, edge_bottom, edge_right))
    return np.vstack(tuple(trimmed_images))


def _rotations(image, num_rotations):
    """
    :param image:
    :return: array of rotated images
    """
    angle = 360 / num_rotations
    rotations = np.empty((num_rotations, image.shape[0], image.shape[1]))
    rotated_image = image
    for i in range(num_rotations):
        rotations[i] = rotated_image
        rotated_image = rotate(rotated_image, angle)
    return rotations


def trim_edges(image, edge_top=2, edge_left=2, edge_bottom=2, edge_right=2):
    rows, columns = _edges(image.shape, edge_top, edge_left, edge_bottom, edge_right)
    return np.delete(np.delete(image, rows, 0), columns, 1)


@memoize
def _edges(shape, edge_top, edge_left, edge_bottom, edge_right):
    height, width = shape
    rows = [i for i in itertools.chain(range(edge_top), range(height-edge_bottom, height))]
    columns = [i for i in itertools.chain(range(edge_left), range(width-edge_right, width))]
    return rows, columns


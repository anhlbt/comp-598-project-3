__author__ = "Charlie, Josh"

from sklearn import datasets
import numpy as np

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


@memoize
def load_raw_data():
    """
    Delegates to load_test_data
    """
    return load_test_data()


def available_datasets():
    """
    """
    return [
        ('iris', load_iris_data),
        ('kaggle images', load_raw_data)
    ]

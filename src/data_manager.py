__author__ = "Charlie"


import numpy as np
from sklearn import datasets


def load_test_data():
	iris = datasets.load_iris()
	return np.matrix(iris.data), np.array(iris.target)

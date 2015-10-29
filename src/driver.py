__author__ = "Charlie"


import importlib
import data_manager
import config
from sklearn.grid_search import GridSearchCV


def load_config():
	"""
	Prompts for user config choice, loads and returns config module.

	@return: config module
	"""
	for i, c in enumerate(config.__all__):
		print("[{}] {}".format(i, c))
	inp = input("Select your config file.\n>>>")
	return importlib.import_module("config." + config.__all__[i])


def load_dataset():
	"""
	Simple data set load implementation.
	This will be moved to a file called data_manager.py that
	should handle loading and vectorizing.

	@return: np.matrix, np,array
	"""
	X, y = data_manager.load_test_data()
	return X, y


def execute_task(configuration, X, y):
	"""
	"""
	for learner, params in configuration.learners.items():
		grid = GridSearchCV(learner, params)
		grid.fit(X, y)
		print(grid.score(X, y))


def main():
	"""
	Main program loop
	"""
	X, y = load_dataset()
	configuration = load_config()
	execute_task(configuration, X, y)


if __name__ == "__main__":
	main()

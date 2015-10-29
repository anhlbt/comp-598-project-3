__author__ = "Charlie"


import importlib
import data_manager
import config
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


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


def build_pipelines(configuration):
	"""
	"""
	pipelines = []
	for learner_tup, learner_params in configuration.learners.items():
		for selector_tup, selector_params in configuration.selectors.items():
			learner_name = learner_tup[0]
			learner = learner_tup[1]
			selector_name = selector_tup[0]
			selector = selector_tup[1]
			pipe = Pipeline([
				(selector_name, selector), 
				(learner_name, learner)
			])
			params = dict(learner_params, **selector_params)
			pipelines.append((pipe, params))
	return pipelines



def execute_configuration(configuration, X, y):
	"""
	"""
	for pipeline, params in build_pipelines(configuration):
		grid = GridSearchCV(pipeline, params)
		grid.fit(X, y)
		print(grid.score(X, y))
	#print(grid.grid_scores_)


def main():
	"""
	Main program loop
	"""
	X, y = load_dataset()
	configuration = load_config()
	execute_configuration(configuration, X, y)


if __name__ == "__main__":
	main()

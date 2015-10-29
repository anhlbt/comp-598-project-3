__author__ = "Charlie"


from sklearn import svm


learners = {
	svm.SVC(): {
		'kernel': ('linear', 'rbf'),
		'C': [1, 10]
	}
}
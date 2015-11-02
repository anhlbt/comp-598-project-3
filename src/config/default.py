__author__ = 'Charlie'

# classifiers
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# feature selection
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
# from sklearn.pipeline import FeatureUnion
# from sklearn.decomposition import PCA


selectors = {
    ('percentile', SelectPercentile()): {
        'percentile__percentile': (1, 5, 25, 50, 100),
        'percentile__score_func': (chi2, f_classif)
    }
}

learners = {
    ('svc', LinearSVC()): {
        'svc__C': (1.0, 10.0)
    },

    ('lr', LogisticRegression()): {
        'lr__penalty': ('l1', 'l2'),
        'lr__C': (1.0, 2.0)
    }
}

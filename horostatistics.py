import random
import difflib
import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report


# =========================================================================== #
# Load data, dropping generic horoscopes

data = pd.read_csv("alldata.csv")
data = data[data["sign"] != "general"]
signs = set(data["sign"].values)


# =========================================================================== #
# Word frequency & Ratcliff-Obershelp difference

keyword_freq = data.groupby("sign")["keywords"].value_counts()


def RO_diff(text1, text2):
    """Ratcliff-Obershelp distance between two pieces of text"""
    return difflib.SequenceMatcher(None, str(text1), str(text2)).ratio()


def avg_RO_diff(corpus, sample=None):
    """Average RO distance between all pieces of text in a corpus"""
    combinations = itertools.combinations(corpus, r=2)
    if sample is not None:
        combinations = random.sample(list(combinations), sample)
    return np.mean([RO_diff(*c) for c in combinations])


def avg_corpus_RO_diff(corpus1, corpus2, sample=None):
    """Average RO distance between two corpuses"""
    pairs = itertools.product(corpus1, corpus2)
    if sample is not None:
        pairs = random.sample(list(pairs), sample)
    return np.mean([RO_diff(*p) for p in pairs])


# =========================================================================== #
# Classification task

# For classification, we only care about the sign and text
data = data[["full_text", "sign"]]
data = data.dropna()

# Convert the text to bag of words representation
vec = TfidfVectorizer(min_df=0.005)
X = vec.fit_transform(data["full_text"].values).toarray()

# Divide the dataset into classes and encode them
pos_polarity = ["aries", "leo", "saggitarius", "gemini", "libra", "aquarius"]
y = data["sign"].apply(lambda x: 1 if x in pos_polarity else 0).values

# Split the data into train and test
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)

# Perform 5-fold CV on the training set to optimize the model params
cross_validation = StratifiedKFold(y_train, n_folds=5)

rf = RandomForestClassifier()
param_grid = {'criterion': ['gini'],
              'n_estimators': [10, 15, 20],
              'max_depth': [1, 2, 3, 4, 5],
              'max_features': [1, 3, 5, 'log2']}

grid_search = GridSearchCV(rf, param_grid = param_grid, cv=cross_validation,
                           scoring="f1")
grid_search.fit(X_train, y_train)

# Use the best model and see how it does on the test set
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_test_pred = best_rf.predict(X_test)

# Print the result
print classification_report(y_test, y_test_pred)
print "\n"
print pd.crosstab(y_test, y_test_pred,
                  rownames=["True"], colnames=["Predicted"],
                  margins=True)
print "\nBest classifier:"
print best_rf

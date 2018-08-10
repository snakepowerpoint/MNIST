import pandas as pd
import numpy as np

from sklearn import datasets
from model import model


iris = datasets.load_iris()
X = iris.data
y = iris.target


clf = LinearSVC(random_state=0)
clf.fit(X, y)
score = clf.decision_function(X)
prediction = clf.predict(X)


score_pd = pd.DataFrame(score)
score_sort = np.sort(score, axis=1)
score_sort = np.argsort(score, axis=1)

score_sort = np.argsort(score)

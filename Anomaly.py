import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
import algo
from sklearn.metrics import classification_report
# TODO make it work
# The idea is to find a very low and very high quality wines as anomaly from regular wines.
# In the seceond part the issue is to find which anomaly is a low-quality and which is high quality.


wine = pd.read_csv('winequality-red.csv', sep=';')
sc = StandardScaler()
X = wine.drop('quality', axis=1)
wine['quality'] = wine['quality'].map({4: 1, 5: 1, 6: 1, 7: 1, 2: -1, 3: -1, 8: -1, 9: -1})
y = wine['quality']
print(y)

X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf = EllipticEnvelope(random_state=0).fit(X_train, y_train)

pred_clf = clf.predict(X_test)
score = pred_clf - y_test

print(score)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# TODO make it work
# The idea is to find a very low and very high quality wines as anomaly from regular wines.
# In the seceond part the issue is to find which anomaly is a low-quality and which is high quality.


wine = pd.read_csv('winequality-red.csv', sep=';')
sc = StandardScaler()
X = wine.drop('quality', axis=1)
ins = 1
outs = 0
wine['quality'] = wine['quality'].map({4: ins, 5: ins, 6: ins, 7: ins, 2: outs, 3: outs, 8: outs, 9: outs})
y = wine['quality']
print(y.value_counts())

X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

print("train set: \n{}".format(y_train.value_counts()))
print("test set: \n{}".format(y_test.value_counts()))

clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print("accuracy for {} neighbours : {}".format(3, accuracy_score(y_test, pred_clf)))
print(classification_report(y_test, pred_clf))

clf = SVC()
param = {
    'C': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
}
grid_svc = GridSearchCV(clf, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
clf = SVC(C=grid_svc.best_params_['C'], gamma=grid_svc.best_params_['gamma'],
          kernel=grid_svc.best_params_['kernel'])
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
print("accuracy for best SVC: {}".format(accuracy_score(y_test, pred_clf)))
print(classification_report(y_test, pred_clf))

clf = MLPClassifier(hidden_layer_sizes=(12, 12, 6, 2), random_state=1, max_iter=3500).fit(
    X_train, y_train)
pred_clf = clf.predict(X_test)
print("accuracy for NN: {}".format(accuracy_score(y_test, pred_clf)))
print(classification_report(y_test, pred_clf))



from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def algo_name(name, lendash=50):
    namelen = len(name)
    lendashmid = int((lendash - namelen) / 2)
    print("-" * lendash)
    print("-" * lendashmid + name + "-" * lendashmid)
    print("-" * lendash)
    print("processing...")


def Stats(pred_clf, y_test):
    raport = classification_report(pred_clf, y_test)
    raportdict = classification_report(pred_clf, y_test, output_dict=True)
    return raportdict, raport


def RandForst(X_train, y_train, X_test, y_test, crossval=False):
    clf = RandomForestClassifier(n_estimators=300)
    algo_name("Random forest Classifier")
    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)
    if crossval == True:
        clf_eval = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
        print("Random forest accuracy  with cross validation: {}, while without CV {}".format(clf_eval.mean(),
                                                                                              accuracy_score(y_test,
                                                                                                             pred_clf)))
    else:
        raportdict, raport = Stats(pred_clf, y_test)
        return raportdict, raport


def sgd(X_train, y_train, X_test, y_test):
    algo_name("Stochastic Gradient Decent Classifier")
    clf = SGDClassifier(penalty=None)
    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)
    raportdict, raport = Stats(pred_clf, y_test)
    return raportdict, raport


def svc(X_train, y_train, X_test, y_test, searchbestparam=False):
    clf = SVC()
    if (searchbestparam == False):
        algo_name("Support Vector Classifier")
    else:
        algo_name("Support Vector Classifier with best parameters")
        param = {
            'C': [1.1, 1.2, 1.3, 1.4],
            'kernel': ['linear', 'rbf'],
            'gamma': [1, 1.1, 1.2, 1.3, 1.4]
        }
        grid_svc = GridSearchCV(clf, param_grid=param, scoring='accuracy', cv=10)
        grid_svc.fit(X_train, y_train)
        clf = SVC(C=grid_svc.best_params_['C'], gamma=grid_svc.best_params_['gamma'],
                  kernel=grid_svc.best_params_['kernel'])
    clf.fit(X_train, y_train)
    pred_clf = clf.predict(X_test)
    raportdict, raport = Stats(pred_clf, y_test)
    return raportdict, raport


def neuralNetwork(X_train, y_train, X_test, y_test):
    algo_name("Multilayer Perceptron classifier")
    clf = MLPClassifier(hidden_layer_sizes=(12, 24, 6), random_state=1, activation='logistic', max_iter=3500).fit(
        X_train, y_train)
    pred_clf = clf.predict(X_test)
    raportdict, raport = Stats(pred_clf, y_test)
    return raportdict, raport


def logisticRegression(X_train, y_train, X_test, y_test):
    algo_name("Logistic regression")
    clf = LogisticRegression(C=1.0).fit(X_train, y_train)
    pred_clf = clf.predict(X_test)
    raportdict, raport = Stats(pred_clf, y_test)
    return raportdict, raport


def DecisionTree(X_train, y_train, X_test, y_test, Bagging=False, AdaBoost=False):
    RANDOM_STATE = 1
    MAX_DEPTH_BOOST = 1
    MIN_SAMPLE_LEAF = 3
    MAX_FEATURES = 50
    clf = DecisionTreeClassifier(min_samples_leaf=MIN_SAMPLE_LEAF,
                                 random_state=RANDOM_STATE,
                                 max_depth=MAX_DEPTH_BOOST).fit(X_train, y_train)
    if Bagging is False and AdaBoost is False:
        algo_name("Decision Tree Classifier")
        pred_clf = clf.predict(X_test)
        raportdict, raport = Stats(pred_clf, y_test)
        return raportdict, raport
    elif Bagging and AdaBoost is False:
        algo_name("Decision Tree Classifier with bagging")
        clf_BaggClass = BaggingClassifier(base_estimator=clf,
                                          random_state=RANDOM_STATE,
                                          max_samples=MAX_FEATURES).fit(X_train, y_train)
        pred_clf = clf_BaggClass.predict(X_test)
        raportdict, raport = Stats(pred_clf, y_test)
        return raportdict, raport
    elif Bagging is False and AdaBoost:
        algo_name("Decision Tree Classifier with Adaboost")
        clf_AdaBoost = AdaBoostClassifier(base_estimator=clf,
                                          random_state=RANDOM_STATE,
                                          algorithm="SAMME",
                                          ).fit(X_train, y_train)
        pred_clf = clf_AdaBoost.predict(X_test)
        raportdict, raport = Stats(pred_clf, y_test)
        return raportdict, raport
    else:
        print("Mistake, please do not start Adaboost and Bagging at the same time")
        return
def kNN(X_train, y_train, X_test, y_test,n=3):
    algo_name("K-Nearest Neighbours")
    clf = KNeighborsClassifier(n_neighbors= n).fit(X_train, y_train)
    pred_clf = clf.predict(X_test)
    raportdict, raport = Stats(pred_clf, y_test)
    return raportdict, raport

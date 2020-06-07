import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import algo


def plot(df, name1, name2):
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(x=name1, y=name2, data=df)
    plt.show()


def results(df, AlgorithmName, BadF1Score, GoodF1Score):
    new_row = {'Algorithm': AlgorithmName, 'Bad wine F1 score': BadF1Score, 'Good wine F1 score': GoodF1Score}
    Results = df.append(new_row, ignore_index=True)
    return Results


print("-" * 50)
print("-" * 50)
print("-" * 6 + "Welcome to the wine quality classifier" + "-" * 6)
print("-" * 50)
print("-" * 50)
while (1):
    usr_wine = input("First choose type of wine. Write 1 for white wine and 2 for red wine: ")
    if usr_wine == "1" or usr_wine == "2":
        break
    else:
        ext = input("type 1 or 2 or to exit press enter: ")
        if ext == "":
            sys.exit()
if (usr_wine == 2):
    wine = pd.read_csv('winequality-red.csv', sep=';')
else:
    wine = pd.read_csv('winequality-white.csv', sep=';')

print("Data set loaded")
print(wine.head())
print(wine.info())

usr_plot = input(" If you want to plot more information/ charts and staff about the set, write 1: ")
if (usr_plot == "1"):
    plot(wine, "quality", "fixed acidity")
    plot(wine, "quality", "volatile acidity")
    plot(wine, "quality", "citric acid")
    plot(wine, "quality", "residual sugar")
    plot(wine, "quality", "chlorides")
    plot(wine, "quality", "free sulfur dioxide")
    plot(wine, "quality", "total sulfur dioxide")

    plt.hist(wine['quality'])
    plt.show()

# pomysl aby podzielic zbior na dwie czesci - wino dobre i złe.

bins = (2, 5.5, 9)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
sns.countplot(wine['quality'])
plt.show()

label_quality = LabelEncoder()
# Bad becomes 0 and good becomes 1
X = wine.drop('quality', axis=1)
y = wine['quality']

wine['quality'] = label_quality.fit_transform(y)
print(wine['quality'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standarize Data
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Prepare DataFrame for results
Results = pd.DataFrame(columns=['Algorithm', 'Bad wine F1 score', 'Good wine F1 score'])

randFrostRsltDict, randFrostRslt = algo.RandForst(X_train, y_train, X_test, y_test)
Results = results(Results, "RandForst", randFrostRsltDict['bad']['f1-score'], randFrostRsltDict['good']['f1-score'])

LogRegRsltDict, LogRegRslt = algo.logisticRegression(X_train, y_train, X_test, y_test)
Results = results(Results, "Log Reg", LogRegRsltDict['bad']['f1-score'], LogRegRsltDict['good']['f1-score'])

SGDRsltDict, SGDRslt = algo.sgd(X_train, y_train, X_test, y_test)
Results = results(Results, "SGD", SGDRsltDict['bad']['f1-score'], SGDRsltDict['good']['f1-score'])

SVCRsltDict, SVCRslt = algo.svc(X_train, y_train, X_test, y_test)
Results = results(Results, "SVC", SVCRsltDict['bad']['f1-score'], SVCRsltDict['good']['f1-score'])

# SVCBestParRsltDict, SVCBestParRslt = algo.svc(X_train, y_train, X_test, y_test, searchbestparam=True)
# Results = results(Results, "SVC Best Param", SVCBestParRsltDict['bad']['f1-score'],
#                   SVCBestParRsltDict['good']['f1-score'])

MLPRsltDict, MLPRslt = algo.neuralNetwork(X_train, y_train, X_test, y_test)
Results = results(Results, "Neural Network", MLPRsltDict['bad']['f1-score'], MLPRsltDict['good']['f1-score'])

DecTreeRsltDict, DecTreeRslt = algo.DecisionTree(X_train, y_train, X_test, y_test)
Results = results(Results, "Decision Tree", DecTreeRsltDict['bad']['f1-score'], DecTreeRsltDict['good']['f1-score'])

DecTreeBaggingRsltDict, DecTreeBaggingRslt = algo.DecisionTree(X_train, y_train, X_test, y_test, Bagging=True)
Results = results(Results, "Decision Tree Bagging", DecTreeBaggingRsltDict['bad']['f1-score'],
                  DecTreeBaggingRsltDict['good']['f1-score'])

DecTreeAdaBoostRsltDict, DecTreeAdaBoostRslt = algo.DecisionTree(X_train, y_train, X_test, y_test, AdaBoost=True)
Results = results(Results, "Decision Tree AdaBoost", DecTreeAdaBoostRsltDict['bad']['f1-score'],
                  DecTreeAdaBoostRsltDict['good']['f1-score'])

kNNRsltDict, kNNRslt = algo.kNN(X_train, y_train, X_test, y_test)
Results = results(Results, "KNeighbors Classifier", kNNRsltDict['bad']['f1-score'],
                      kNNRsltDict['good']['f1-score'])

print(Results)
print("Best F1 score for good wines: {} dla algorytmu {}".format(Results['Good wine F1 score'].max(),
                                                                 Results['Algorithm'][
                                                                     Results['Good wine F1 score'].idxmax()]))
print("Best F1 score for bad wines: {} dla algorytmu {}".format(Results['Bad wine F1 score'].max(),
                                                                Results['Algorithm'][
                                                                    Results['Bad wine F1 score'].idxmax()]))
algo.RandForst(X_train, y_train, X_test, y_test, crossval=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


#TODO make it work
# The idea is to find a very low and very high quality wines as anomaly from regular wines.
# In the seceond part the issue is to find which anomaly is a low-quality and which is high quality.




def estimate_gaussian(X):
    mu = []
    sigma2 = []
    for i in range(0,X.shape[1]):
        mu.append(np.mean(X[:, i]))
        sigma2.append(np.var(X[:, i]))
    return mu, sigma2

def select_threshold(pval, yval):
    space = np.array(np.linspace(pval.min(),pval.max(),1000))
    best_eps = space[0]
    eps = space[0]
    pred = np.zeros((len(yval), 1))
    for k in range(len(yval)):
        for j in range (0,pval.shape[1]):
            if pval[k,j] > eps:
                pred[k] = 1
    best_f1 = f1_score(yval, pred,average='micro')
    for i in range(len(space)):
        eps = space[i]
        pred = np.zeros((len(yval),1))
        for k in range(len(yval)):
            for j in range(0,pval.shape[1]):
                if pval[k, j] < eps:
                    pred[k] = 1
        f1 = f1_score(yval, pred,average='micro')
        if f1 > best_f1:
            best_eps = eps
            best_f1 = f1
    return best_eps, best_f1



wine = pd.read_csv('winequality-red.csv', sep=';')



sc = StandardScaler()
X = wine.drop('quality', axis=1)
y = wine['quality']
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
plt.figure(1)
plt.hist(wine['quality'])
plt.title("Histogram for quality")

mu, sigma2 = estimate_gaussian(X_train)
print("Mean is {} and sigma2 is {}".format(mu, sigma2))

p = np.zeros((X.shape[0], X.shape[1]))
pval = np.zeros((X_test.shape[0], X_test.shape[1]))
for i in range(0,X.shape[1]):
    p[:, i] = stats.norm.pdf(X[:, i], mu[i], np.sqrt(sigma2[i]))
    pval[:, i] = stats.norm.pdf(X_test[:, i], mu[i], np.sqrt(sigma2[i]))



best_eps, best_F1 = select_threshold(pval,y_test)
print(best_eps, best_F1 )

anomal = np.where(pval < best_eps)
anomal = list(dict.fromkeys(anomal[0]))

print(anomal)
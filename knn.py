#Mahmoud Mohamed Amr 20176027
#Hady Raed 20175019
import numpy as np
import pandas as pd
from scipy.stats import mode


def eucledian(p1, p2):
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    return dist


def getNear(classes):
    near = mode(classes)
    return near.mode[0]


def knn(xtrain, y, xtest, k):
    final = []
    for item in xtest:

        points = []
        for j in range(len(xtrain)):
            distances = eucledian(np.array(xtrain[j, :]), item)
            points.append(distances)
        points = np.array(points)
        dist = np.argsort(points)[:k]

        kclasses = y[dist]
        final.append(getNear(kclasses))
    return final


def calAccurancy(x, y, k):
    count = 0
    for i in range(0, len(x)):
        if x[i] == y[i]:
            count += 1
    print("K = " + str(k))
    print("Number of correctly classified instances: %d Total number of instances : %d " % (count, len(x)))
    print("Accuracy :" + str(count / len(x)))
    print("")


data = pd.read_csv('TrainData.txt', header=None)
dataTest = pd.read_csv('TestData.txt', header=None)
ytest = list(dataTest[8])
ytrain = list(data[8])

del data[8]
del dataTest[8]

for i in range(0, 8):
    data[i] = pd.to_numeric(data[i])
xtest = dataTest.to_numpy()

for i in range(0, 8):
    dataTest[i] = pd.to_numeric(dataTest[i])
xtrain = data.to_numpy()

classes = {
    'VAC': 0, 'ME3': 1, 'ME2': 2, 'ME1': 3, 'CYT': 4, 'MIT': 5, 'POX': 6, 'NUC': 7, 'EXC': 8, 'ERL': 9

}
newYtrian = []
for i in ytrain:
    newYtrian.append(classes[i])
newYtrian = np.array(newYtrian)

newYtest = []
for i in ytest:
    newYtest.append(classes[i])
newYtest = np.array(newYtest)

###reading data end

k = [1, 3, 5, 7, 9]

for i in k:
    y_pred = knn(xtrain, newYtrian, xtest, i)
    calAccurancy(y_pred, newYtest, i)

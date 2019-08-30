# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy

def verify(trueLabels, kLabels):
    mapp = {k: k for k in numpy.unique(kLabels)}
    for k in numpy.unique(kLabels):
        k_mapping = numpy.argmax(numpy.bincount(kLabels[trueLabels==k]))
        mapp[k] = k_mapping
    predictions = [mapp[label] for label in trueLabels]
    print(mapp)
    return mapp, predictions

data,label =load_digits(return_X_y=True)
clf = KMeans(n_clusters=10,random_state=42)
clf.fit(data)
y = clf.predict(data)
trainMapping , trainPredictions = verify(label,y )

print('Accuracy: {}'.format(accuracy_score(y, trainPredictions)))

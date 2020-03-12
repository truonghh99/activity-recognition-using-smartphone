import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn import svm

path_1 = 'dataset/train.csv'
path_2 = 'dataset/test.csv'

df_1 = pd.read_csv(path_1)
df_2 = pd.read_csv(path_2)

df = pd.concat([df_1, df_2])

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

x_train = train.iloc[:,0:562]
y_train = train['Activity']
x_test = test.iloc[:,0:562]
y_test = test['Activity']

clf = svm.SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from mpl_toolkits import mplot3d
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report    
from sklearn import tree
from sklearn.metrics import confusion_matrix 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

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
clf = ExtraTreesClassifier()
clf = clf.fit(x_train, y_train)

x_test = test.iloc[:,0:562]
y_test = test['Activity']
y_pred = clf.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn import svm
from sklearn.metrics import confusion_matrix 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns


path = 'dataset/data.csv'
df = pd.read_csv(path)
target_labels =  ['LAYING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS','SITTING']
df['Activity'] = df["Activity"].map({"LAYING": 0, "STANDING": 1, 'WALKING': 2, 'WALKING_DOWNSTAIRS': 3, 'WALKING_UPSTAIRS': 4, 'SITTING': 5})

# assign 66% of each type to training dataset
num_type = [0,0,0,0,0,0]
for index, row in df.iterrows():
    num_type[int(row[57])] += 1

num_test = [int(i * 0.66) for i in num_type]

train, test = [],[]

for index, row in df.iterrows():
    num_test[int(row[57])] -= 1
    if (num_test[int(row[57])] >= 0):
        train.append(row)
    else:
        test.append(row)
 
train = pd.DataFrame(train)
test = pd.DataFrame(test)

x_train = train.iloc[:,0:562]
y_train = train['Activity']
x_test = test.iloc[:,0:562]
y_test = test['Activity']

# Decision Tree
clf_dt = DecisionTreeClassifier(min_impurity_decrease = 0.002)
clf_dt = clf_dt.fit(x_train, y_train)
y_pred_dt = clf_dt.predict(x_test)

# SVM 
clf_svm = svm.SVC()
clf_svm.fit(x_train, y_train)
y_pred_svm = clf_svm.predict(x_test)

# Neural Network
clf_nn = MLPClassifier()
clf_nn = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)


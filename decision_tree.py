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

path = 'dataset/data.csv'
df = pd.read_csv(path)

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

x_train = train.iloc[:,0:562]
y_train = train['Activity']
x_test = test.iloc[:,0:562]
y_test = test['Activity']
clf = DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix on training set: ")
print(confusion_matrix(y_train, clf.predict(x_train)))
print("Confusion matrix on testing set: ")
print(confusion_matrix(y_test, y_pred))

# visualize decision tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=['0','1','2','3','4'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree.png')
Image(graph.create_png())
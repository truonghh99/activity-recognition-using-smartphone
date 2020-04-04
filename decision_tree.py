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
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns

path = 'dataset/data.csv'
df = pd.read_csv(path)
target_labels =  ['LAYING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS','SITTING']
df['Activity'] = df["Activity"].map({"LAYING": 0, "STANDING": 1, 'WALKING': 2, 'WALKING_DOWNSTAIRS': 3, 'WALKING_UPSTAIRS': 4, 'SITTING': 5})
print(df['Activity'])

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

x_train = train.iloc[:,0:562]
y_train = train['Activity']
x_test = test.iloc[:,0:562]
y_test = test['Activity']
clf = DecisionTreeClassifier(min_impurity_decrease = 0.002)

clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion matrix on training set: ")
print(confusion_matrix(y_train, clf.predict(x_train)))
print("Confusion matrix on testing set: ")
print(confusion_matrix(y_test, y_pred))	


"""
# visualize decision tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,class_names=target_labels)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('decision_tree_with_min_impurity.png')
Image(graph.create_png())
"""

def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize):
    y_score = clf.predict_proba(X_test)
    print(y_score)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Activity Recognition Decision Tree Model ROC Curve')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = {}) for label {}'.format(roc_auc[i], target_labels[i]))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(clf, x_test, y_test, n_classes=6, figsize=(16, 10))

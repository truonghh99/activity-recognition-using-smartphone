import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score 
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import seaborn as sns
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

clf_dt = DecisionTreeClassifier(min_impurity_decrease = 0.002)
clf_svm = svm.SVC()
clf_nn = MLPClassifier()

clf_total = [clf_dt, clf_svm, clf_nn]
clf_names = ["Decision Tree", "SVM", "Neural Network"]

for clf in clf_total:
	clf.fit(x_train, y_train)
	#y_pred = clf.predict(x_test)
	#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def plot_label(label, X_test, y_test):
	fig, ax = plt.subplots(figsize=(10,10))
	ax.plot([0, 1], [0, 1], 'k--')
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title('Activity Recognition ROC Curve for {}'.format(target_labels[label]))
	index = 0
	for clf in clf_total:
		y_score = 0
		if (index == 1):
			y_score = clf.decision_function(X_test)
		else:
			y_score = clf.predict_proba(X_test)
		y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
		fpr, tpr, _ = roc_curve(y_test_dummies[:, label], y_score[:, label])
		roc_auc = auc(fpr, tpr)
		ax.plot(fpr, tpr, label='ROC curve (area = {}) using {}'.format(roc_auc, clf_names[index]))
		index += 1
	ax.legend(loc="best")
	ax.grid(alpha=.4)
	sns.despine()
	plt.show()

for i in range(6):
	plot_label(i, x_test, y_test)
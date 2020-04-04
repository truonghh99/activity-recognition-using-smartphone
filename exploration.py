import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits import mplot3d


path = 'dataset/data.csv'
df = pd.read_csv(path)

target_labels =  ['LAYING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS','SITTING']

#distribution
pd.Series(df['Activity']).value_counts().plot(kind = 'bar', rot = 0)
plt.show()


#df['Activity'] = df["Activity"].map({"LAYING": 1, "STANDING": 2, 'WALKING': 3, 'WALKING_DOWNSTAIRS': 4, 'WALKING_UPSTAIRS': 5, 'SITTING': 6})

# Plot 3 dimensional figures

toDraw = [['tBodyAcc-correlation()-Y,Z','tGravityAcc-max()-X','tGravityAccMag-arCoeff()4'],['tGravityAccMag-arCoeff()4', 'fBodyAccMag-sma()', 'tGravityAcc-max()-X']]
for i in toDraw:
	ax = plt.axes(projection='3d')
	ax.set_xlabel(i[0])
	ax.set_ylabel(i[1])
	ax.set_zlabel(i[2])
	x = df[i[0]]
	y = df[i[1]]
	z = df[i[2]]
	ax.scatter3D(x, y, z, alpha = 0.5, 
				c = df['Activity'].map({"LAYING": 1, "STANDING": 2, 'WALKING': 3, 'WALKING_DOWNSTAIRS': 4, 'WALKING_UPSTAIRS': 5, 'SITTING': 6}), 
				label = df['Activity'])
	plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits import mplot3d
from itertools import combinations 

path = 'dataset/data.csv'
df = pd.read_csv(path)

target_labels =  ['LAYING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS','SITTING']

#distribution
pd.Series(df['Activity']).value_counts().plot(kind = 'bar', rot = 0)
#plt.show()


#df['Activity'] = df["Activity"].map({"LAYING": 1, "STANDING": 2, 'WALKING': 3, 'WALKING_DOWNSTAIRS': 4, 'WALKING_UPSTAIRS': 5, 'SITTING': 6})

#df = df[df['Activity'] != "LAYING"]
#df = df[df['Activity'] != "WALKING_DOWNSTAIRS"]
#df = df[df['Activity'] != "WALKING_UPSTAIRS"]
#df = df[df['Activity'] != "WALKING"]
#df = df[df['Activity'] != "STANDING"]
#df = df[df['Activity'] != "SITTING"]

# Plot 3 dimensional figures
toDraw = list(combinations(df.columns[35:508], 3))
for i in toDraw:
	ax = plt.axes(projection='3d')
	ax.set_xlabel(i[0])
	ax.set_ylabel(i[1])
	ax.set_zlabel(i[2])
	x = df[i[0]]
	y = df[i[1]]
	z = df[i[2]]
	ax.scatter3D(x, y, z, alpha = 0.5, 
				c = df['Activity'].map({"LAYING": 'red',
										"STANDING": 'blue', 
										'WALKING': 'green', 
										'WALKING_DOWNSTAIRS': 'purple', 
										'WALKING_UPSTAIRS': 'orange',
										'SITTING': 'yellow'}), 
				label = df['Activity'])
	plt.show()
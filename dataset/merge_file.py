import os
import glob
import pandas as pd

path_1 = 'train.csv'
path_2 = 'test.csv'
combined_csv = pd.concat([pd.read_csv(path_1), pd.read_csv(path_2)])
combined_csv.to_csv("data.csv", index=False, encoding='utf-8-sig')

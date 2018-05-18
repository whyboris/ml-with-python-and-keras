print('try the titanic Kaggle competition')

import pandas as pd 

titanic_path = '../titanic/data.csv'

dataset = pd.read_csv(titanic_path, quotechar='"')

print(dataset.iloc[0:5, 0:10])


print('try the titanic Kaggle competition')

import pandas as pd 

titanic_path = '../titanic/data.csv'

dataset = pd.read_csv(titanic_path, quotechar='"')

# cleaned data set looks like this (first seven rows):
# print(dataset.iloc[0:7, 0:10])
# print(dataset.head(7))
# print(dataset.tail(7))

from keras.utils.np_utils import to_categorical

# result = []

# # categorical data:
# for i in range(1, 6):
#   lol = dataset.iloc[0:7, i:i+1]
#   # print(to_categorical(lol))
#   # print(to_categorical(lol).shape)
#   result.append(to_categorical(lol)[3])

# # numerical data
# for i in range(6, 9):
#   lol = dataset.iloc[0:7, i:i+1]
#   print(lol.max())
#   print(lol)
#   lol = lol / lol.max()

# print(result)


# get Pclass ready !
pclass = to_categorical(dataset['Pclass'].tolist())
# get Sex ready !
sex = to_categorical(dataset['Sex'].tolist())
# get Parch ready !
parch = to_categorical(dataset['Parch'].tolist())
# get Cabin ready !
cabin = to_categorical(dataset['Cabin'].tolist())
# get Embarked ready !
embarked = to_categorical(dataset['Embarked'].tolist())

# get fare list ready !
fare = dataset['Fare'].tolist()
max_fare = max(fare)
fare = [x / max_fare for x in fare]

# get age ready !
age = dataset['Age'].tolist()
max_age = max(age)
age = [x / max_age for x in age]

# get number of sib & spouses ready !
sib_sp = dataset['SibSp'].tolist()
max_sib_sp = max(sib_sp)
sib_sp = [x / max_sib_sp for x in sib_sp]

# survived
survived = dataset['Survived'].tolist()

import numpy as np

train_data = []

train_results = np.array(survived)

train_results = train_results[:5]

# create tensor
for i in range(0, 5):
    # train_data = np.append(train_data, np.array([pclass[i], sex[i], parch[i], cabin[i], embarked[i], fare[i], age[i], sib_sp[i]]), axis=0)
    train_data.append(np.array([
                                pclass[i], 
                                sex[i], 
                                parch[i], 
                                cabin[i], 
                                embarked[i], 
                                np.array(fare[i]), 
                                np.array(age[i]), 
                                np.array(sib_sp[i])
                              ]))
    
    # train_results.append(survived[i]))



train_data = np.array(train_data).reshape(5, 8)

print(np.array(train_results))
print(np.array(train_results).shape)

train_results = np.array(train_results)

# print(train_results.shape)

print('shape is:')
print(train_data.shape)
print('each row is shaped:')
print(train_data[1].shape)
print('example row looks like this:')
print(train_data[1])

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(8,))) # input shape batch dimension should NOT be included
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

#temp hack
print('expected results:')
print(train_results)
print(train_results.shape)

model.fit(train_data, train_results, epochs=5, batch_size=1)



# test_mse_score, test_mae_score = model.evaluate(train_data, survived)

# print('Mean attribute error: $', test_mae_score.round(3) * 1000)


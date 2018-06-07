print(' ')
print(' ')
print('try the titanic Kaggle competition')
print(' ')
print(' ')

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

print(' ')
print(' ')

print(dataset[0:5])

print('num of items:')
print(len(dataset))

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# print(dataset.as_matrix(columns=['Pclass']))

# labelencoder_X_1 = LabelEncoder()
# pclass2 = labelencoder_X_1.fit_transform(dataset['Pclass'])
# print(pclass2)

# onehotencoder = OneHotEncoder(categorical_features=[1])
# pclass3 = onehotencoder.fit_transform(pclass2).toarray()
# print(pclass3)

def get_column(name):
    return dataset.as_matrix(columns=[name])

# print(get_column('Pclass'))

print(' ')
print(' ')
print(' ')
print(' ')

# get Pclass ready !
# pclass = to_categorical(dataset['Pclass'].tolist())
pclass = get_column('Pclass')
# get Sex ready !
# sex = to_categorical(dataset['Sex'].tolist())
sex = get_column('Sex')
# get Parch ready !
# parch = to_categorical(dataset['Parch'].tolist())
parch = get_column('Parch')
# get Cabin ready !
# cabin = to_categorical(dataset['Cabin'].tolist())
cabin = get_column('Cabin')
# get Embarked ready !
# embarked = to_categorical(dataset['Embarked'].tolist())
embarked = get_column('Embarked')

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

# training set size !!!
t_size = 700

train_data = []

train_results = np.array(survived)
train_results = train_results[:t_size]

val_results = np.array(survived)
val_results = val_results[t_size:len(val_results)]

# create tensor
for i in range(0, t_size):
    # train_data = np.append(train_data, np.array([pclass[i], sex[i], parch[i], cabin[i], embarked[i], fare[i], age[i], sib_sp[i]]), axis=0)
    train_data.append(np.array([
                                pclass[i], 
                                sex[i], 
                                parch[i], 
                                cabin[i], 
                                embarked[i], 
                                np.array(fare[i]*5), 
                                np.array(age[i]*5), 
                                np.array(sib_sp[i])
                              ]))
    
    # train_results.append(survived[i]))

val_data = []

# crap DUPLICATE CODE
for i in range(t_size, len(dataset)):
    # train_data = np.append(train_data, np.array([pclass[i], sex[i], parch[i], cabin[i], embarked[i], fare[i], age[i], sib_sp[i]]), axis=0)
    val_data.append(np.array([
                                pclass[i], 
                                sex[i], 
                                parch[i], 
                                cabin[i], 
                                embarked[i], 
                                np.array(fare[i]*5), 
                                np.array(age[i]*5), 
                                np.array(sib_sp[i])
                              ]))


train_data = np.array(train_data).reshape(t_size, 8)

val_dataset_len = len(dataset) - t_size
print(val_dataset_len)
val_data = np.array(val_data).reshape(val_dataset_len, 8)

# print(np.array(train_results))
# print(np.array(train_results).shape)

train_results = np.array(train_results)
val_results = np.array(val_results)

# print(train_results.shape)

# print('shape is:')
# print(train_data.shape)
# print('each row is shaped:')
# print(train_data[1].shape)
# print('example row looks like this:')
# print(train_data[1])

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(8,))) # input shape batch dimension should NOT be included
model.add(layers.Dropout(0.1))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


x_val = val_data
y_val = val_results

# print('expected results:')
# print(train_results)
# print(train_results.shape)

# print(train_data[0])
# print(train_data[1])
# print(train_data[2])

# print(x_val[0])
# print(x_val[1])
# print(x_val[2])

history = model.fit(train_data, train_results, 
                    epochs=600, batch_size=t_size, 
                    validation_data=(x_val, y_val))


# test_mse_score, test_mae_score = model.evaluate(train_data, survived)

# print('Mean attribute error: $', test_mae_score.round(3) * 1000)



import matplotlib.pyplot as plt  # pylint disable=E0401

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

# don't pring out results of the first number epochs
trim = 10

plt.plot(epochs[trim:], loss[trim:], 'bo', label='Training loss')
plt.plot(epochs[trim:], val_loss[trim:], 'b', label='Validation loss')
plt.title('Titanic training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

accuracy = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs[trim:], accuracy[trim:], 'bo', label='Accuracy')
plt.plot(epochs[trim:], val_acc[trim:], 'b', label='Validation accuracy')
plt.title('Titanic training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

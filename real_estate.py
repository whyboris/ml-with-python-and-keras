print('Real estate regression')

from hack import hack

hack()


from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)

print(test_data.shape)

# print(train_targets.shape)

import numpy as np 

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

from keras import models
from keras import layers


def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

  return model

# run k-fold because the data set is small
# we would then print the performance over epochs
# determine when model starts overfitting
# and then perform a regular run (see below)

# k = 4
# num_val_samples = len(train_data) // k
# num_epochs = 200
# all_scores = []

# for i in range(k):
#   print('processing fold: ', i)
#   val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#   val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

#   partial_train_data = np.concatenate([train_data[:i * num_val_samples],
#                                        train_data[(i + 1) * num_val_samples:]], axis=0)

#   partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
#                                           train_targets[(i + 1) * num_val_samples:]], axis=0)

#   model = build_model()

#   model.fit(partial_train_data, partial_train_targets,
#             epochs=num_epochs, batch_size=1, verbose=0)

#   val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

#   all_scores.append(val_mae)

# print(all_scores)

# graph output to see when overfitting starts

model = build_model()
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print('Mean attribute error: $', test_mae_score.round(3) * 1000)








# %%
import numpy as np # numpy for matrix operations
import pandas as pd # for reading csv files and manipulate with them
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../"))

# %%
# import dataset 
data = pd.read_csv("../input/breast_cancer_data.csv")
print(data.head())
# ['.config', 'breast_cancer_data.csv', 'sample_data']
#          id diagnosis  radius_mean  texture_mean  perimeter_mean  area_mean  \
# 0    842302         M        17.99         10.38          122.80     1001.0   
# 1    842517         M        20.57         17.77          132.90     1326.0   
# 2  84300903         M        19.69         21.25          130.00     1203.0   
# 3  84348301         M        11.42         20.38           77.58      386.1   
# 4  84358402         M        20.29         14.34          135.10     1297.0   

#    smoothness_mean  compactness_mean  concavity_mean  concave points_mean  \
# 0          0.11840           0.27760          0.3001              0.14710   
# 1          0.08474           0.07864          0.0869              0.07017   
# 2          0.10960           0.15990          0.1974              0.12790   
# 3          0.14250           0.28390          0.2414              0.10520   
# 4          0.10030           0.13280          0.1980              0.10430   

#    ...  texture_worst  perimeter_worst  area_worst  smoothness_worst  \
# 0  ...          17.33           184.60      2019.0            0.1622   
# 1  ...          23.41           158.80      1956.0            0.1238   
# 2  ...          25.53           152.50      1709.0            0.1444   
# 3  ...          26.50            98.87       567.7            0.2098   
# 4  ...          16.67           152.20      1575.0            0.1374   

#    compactness_worst  concavity_worst  concave points_worst  symmetry_worst  \
# 0             0.6656           0.7119                0.2654          0.4601   
# 1             0.1866           0.2416                0.1860          0.2750   
# 2             0.4245           0.4504                0.2430          0.3613   
# 3             0.8663           0.6869                0.2575          0.6638   
# 4             0.2050           0.4000                0.1625          0.2364   

#    fractal_dimension_worst  Unnamed: 32  
# 0                  0.11890          NaN  
# 1                  0.08902          NaN  
# 2                  0.08758          NaN  
# 3                  0.17300          NaN  
# 4                  0.07678          NaN  

# [5 rows x 33 columns]

# %%
# As wee see, unnamed has null values and id is uniqie (no repetition),
# it means that these values do not have value for us. To drop them
# we use drop function with names of columns. axis = 1 means drop columns,
# inplace = True means do operation and return the resuling dataframe
data.drop(['Unnamed: 32',"id"], axis=1, inplace=True) 

print("Diagnosis values", data.diagnosis.values)

# We need to convert M to 1 and B to 0 for AI model to understand the values correctly
listForDiagnosis = list()

for i in range(len(data.diagnosis)):
    str = data.diagnosis[i]
    if str == "M":
        listForDiagnosis.append(1)
    else:
        listForDiagnosis.append(0)

data.diagnosis = listForDiagnosis

print("New diagnosis values", data.diagnosis.values)

y = data.diagnosis.values
x_non_normalized = data.drop(['diagnosis'], axis=1)


# %% 
# min max normalization gave lower accuracy that z-score standardization
# x = (x_non_normalized -np.min(x_non_normalized))/(np.max(x_non_normalized)-np.min(x_non_normalized)).values

# z-score standardization
scaller = StandardScaler()
x = scaller.fit_transform(x_non_normalized)
print(f"x.shape {x.shape}, x: {x}")
# x.shape (569, 30), x: [[ 1.09706398 -2.07333501  1.26993369 ...  2.29607613  2.75062224
#    1.93701461]
#  [ 1.82982061 -0.35363241  1.68595471 ...  1.0870843  -0.24388967
#    0.28118999]
#  [ 1.57988811  0.45618695  1.56650313 ...  1.95500035  1.152255
#    0.20139121]
#  ...
#  [ 0.70228425  2.0455738   0.67267578 ...  0.41406869 -1.10454895
#   -0.31840916]
#  [ 1.83834103  2.33645719  1.98252415 ...  2.28998549  1.91908301
#    2.21963528]
#  [-1.80840125  1.22179204 -1.81438851 ... -1.74506282 -0.04813821
#   -0.75120669]]

# %%
# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

print(f'x_train {x_train.shape}')
print(f'x_test {x_test.shape}')
print(f'y_train {y_train.shape}')
print(f'y_test {y_test.shape}')
# x_train (455, 30)
# x_test (114, 30)
# y_train (455,)
# y_test (114,)

# Creating ANN model with sequential layers
# %%
classifier = Sequential()

# First hidden layer
# Layers are dense layer, which means that every neuron will be connected to every neuron
# in the next and in the previous layers.
# Input size will be 30 neurons (first layer) and the second layer (first hidden layer)
# size will be 16 neuron. Activation function is relu - rectified linear unit.
# kernel_initializer is the property that you set when you firstly initialize the random weights
# of ANN. "random_normal" generates random numbers within a normal distribution.
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=30))

# second hidden layer
# The amount of neruon is 6 and activation function is relu
classifier.add(Dense(6, activation='relu', kernel_initializer='random_normal'))

# third hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='random_normal'))

# Fourth hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='random_normal'))

# output layer
# It is again a dense layer with one neuron, because our model needs output in form of 1 or 0,
# Malignant or Bening. Activation function is sigmoid because this function transforms the value to be
# in the range of 0 or 1. Formula is: 1/(1 + e^-x).
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

# compile the ANN. Prepare ANN for the work. 
# optimizer='adam' means that this ANN will use Adam algorithm. It is an algorithm
# that uses Stochastic Gradient Descent algorithm. It is good for large datasets.
# loss='binary_crossentropy' is the algorithm of finding loss (error or, we can say difference between
# predicted and real outputs). 'binary_crossentropy' is the best for finding errors between binary 
# (True or False) outputs.
# metrics = ['accuracy'] helps to evaluate a model. It creates two variables, count and count
# and this metric counts how many times predicted output(label) was the same as real output(label).
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

# %%
# Train the model.
# We are passing train inputs and train outputs.
# Also we give batch_size (amount of samples that our algortihm will use to change the weights).
# And we give epochs number. It is the amount of times how often the ANN will go forward and backward.
# validation_split is the share of your dataset used for validation of your ANN model.
resultFit = classifier.fit(x_train, y_train, validation_split=0.1, batch_size=20, epochs=150)
# Epoch 1/150
# 21/21 [==============================] - 1s 8ms/step - loss: 0.6920 - accuracy: 0.5990 - val_loss: 0.6886 - val_accuracy: 0.6957
# Epoch 2/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.6887 - accuracy: 0.6210 - val_loss: 0.6826 - val_accuracy: 0.6957
# Epoch 3/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.6813 - accuracy: 0.6210 - val_loss: 0.6689 - val_accuracy: 0.6957
# Epoch 4/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.6608 - accuracy: 0.6284 - val_loss: 0.6315 - val_accuracy: 0.7609
# Epoch 5/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.6066 - accuracy: 0.8191 - val_loss: 0.5411 - val_accuracy: 0.8913
# Epoch 6/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.4914 - accuracy: 0.9609 - val_loss: 0.3800 - val_accuracy: 0.9565
# Epoch 7/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.3258 - accuracy: 0.9682 - val_loss: 0.2272 - val_accuracy: 0.9565
# Epoch 8/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.1985 - accuracy: 0.9682 - val_loss: 0.1465 - val_accuracy: 0.9565
# Epoch 9/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.1359 - accuracy: 0.9707 - val_loss: 0.1120 - val_accuracy: 0.9783
# Epoch 10/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.1079 - accuracy: 0.9756 - val_loss: 0.0942 - val_accuracy: 0.9783
# Epoch 11/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.0944 - accuracy: 0.9756 - val_loss: 0.0861 - val_accuracy: 0.9783
# Epoch 12/150
# 21/21 [==============================] - 0s 2ms/step - loss: 0.0848 - accuracy: 0.9756 - val_loss: 0.0804 - val_accuracy: 0.9783
# Epoch 13/150
# ...
# Epoch 149/150
# 21/21 [==============================] - 0s 2ms/step - loss: 8.2595e-04 - accuracy: 1.0000 - val_loss: 0.0053 - val_accuracy: 1.0000
# Epoch 150/150
# 21/21 [==============================] - 0s 2ms/step - loss: 8.1008e-04 - accuracy: 1.0000 - val_loss: 0.0053 - val_accuracy: 1.0000

# %%
# see the acuracy of the model
eval_model=classifier.evaluate(x_train, y_train)
print("Accuracy of a model", eval_model)
# loss: 0.0012 - accuracy: 1.0000
# Accuracy of a model [0.0012488440843299031, 1.0]

# %%
# predict the outputs of unseen dataset
fromAnnYPred = classifier.predict(x_test)
print(f"Type: {type(fromAnnYPred)} \nPredicted output of unseen dataset {fromAnnYPred}")

for i in range(len(fromAnnYPred)):
    if fromAnnYPred[i] > 0.5:
        fromAnnYPred[i] = 1
    else:
        fromAnnYPred[i] = 0

print(f"Type: {type(fromAnnYPred)} \nPredicted output (changed) of unseen dataset {fromAnnYPred}")
# Type: <class 'numpy.ndarray'> 
# Predicted output of unseen dataset [[6.5983779e-04]
#  [1.0000000e+00]
#  [1.0000000e+00]
#  [8.8799608e-09]
#  [9.0896363e-10]
#  [1.0000000e+00]
#  [1.0000000e+00]
#  [1.0000000e+00]
#  [1.2043638e-03]
#  [1.8054527e-10]
#  [7.7732368e-03]
#  [1.0000000e+00]
#  [2.5111581e-07]
#  [1.0000000e+00]
#  [3.1538310e-09]
#  [1.0000000e+00]
#  [3.2399068e-08]
#  [3.7037856e-10]
#  [7.2307147e-12]
#  [1.0000000e+00]
#  [2.2962403e-04]
#  [2.3761550e-06]
#  [1.0000000e+00]
# ...
#  [1.]
#  [0.]
#  [0.]
#  [1.]]

# %%
# create a confusion matrix of unseen dataset
cm = confusion_matrix(y_test, fromAnnYPred)
print(cm)
# show the accuracy score of unseen dataset
print(f" Accuracy score of testing dataset: {accuracy_score(y_test, fromAnnYPred)}")
# [[70  1]
#  [ 1 42]]
#  Accuracy score of testing dataset: 0.9824561403508771

# %%
print(f"Type of resultFit {type(resultFit)}, resultFit: {resultFit}")
history_dict = resultFit.history
print(history_dict.keys())
# Type of resultFit <class 'keras.callbacks.History'>, resultFit: <keras.callbacks.History object at 0x00000280A092EE60>
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


# %%
# Plot accuracy of a model with recpect to number of epochs.
plt.plot(history_dict['accuracy'])
plt.plot(history_dict['val_accuracy'])
plt.title('Accuracy of a model')
plt.ylabel('Accuracy percent')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%
# Plot loss of a model with respect to number of epochs
plt.plot(history_dict['loss'])
plt.plot(history_dict['val_loss'])
plt.title('Loss of a model')
plt.ylabel('Loss function')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# %%

# %%
# Rasul Osmanbayli. 24/01/2023
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
print(os.listdir("./"))

# import dataset 
data = pd.read_csv("./input/breast_cancer_data.csv")
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

print("Diagnosis values", data.diagnosis.values)

y = data.diagnosis.values
x_non_normalized = data.drop(['diagnosis'], axis=1)


# %% 
# min max normalization gave 96%-97% accuracy (test-train)
# x = (x_non_normalized -np.min(x_non_normalized))/(np.max(x_non_normalized)-np.min(x_non_normalized)).values

# z-score standardization gave 97%-98% accuracy (test-train)
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


# Change places of rows and columns with transport (T) function
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print(f'x_train {x_train.shape}')
print(f'x_test {x_test.shape}')
print(f'y_train {y_train.shape}')
print(f'y_test {y_test.shape}')

# Creating ANN model with sequential layers
# %%
classifier = Sequential()

# first hidden layer
# Layer are dense layer, which means that every neuron will be connected to every neuron
# in the next and in the previous layers.
# Input size will be 30 neurons (first layer) and the second layer (first hidden layer)
# size will be 16 neuron. Activation function is relu - rectified linear unit.
# kernel_initializer is the property that you set when you firstly initialize the random weights
# of ANN. "random_normal" generates random normal within a normal distribution.
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=30))

# second hidden layer
# The amount of neruon is 16 (the same as the previous) and activation function is relu
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

# third hidden layer
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

# Fourth hidden layer
classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal'))

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
resultFit = classifier.fit(x_train.T, y_train.T, validation_split=0.1, batch_size=10, epochs=150)
# Epoch 1/150
# 41/41 [==============================] - 0s 4ms/step - loss: 0.1404 - accuracy: 0.9878 - val_loss: 0.3019 - val_accuracy: 0.9348
# Epoch 2/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0703 - accuracy: 0.9902 - val_loss: 0.2135 - val_accuracy: 0.9348
# Epoch 3/150
# 41/41 [==============================] - 0s 2ms/step - loss: 0.0403 - accuracy: 0.9927 - val_loss: 0.1857 - val_accuracy: 0.9565
# Epoch 4/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0208 - accuracy: 0.9951 - val_loss: 0.1461 - val_accuracy: 0.9565
# Epoch 5/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0205 - accuracy: 0.9951 - val_loss: 0.1402 - val_accuracy: 0.9783
# Epoch 6/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0151 - accuracy: 0.9976 - val_loss: 0.1240 - val_accuracy: 0.9565
# Epoch 7/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0139 - accuracy: 0.9976 - val_loss: 0.1268 - val_accuracy: 0.9565
# Epoch 8/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0127 - accuracy: 0.9976 - val_loss: 0.1202 - val_accuracy: 0.9565
# Epoch 9/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0127 - accuracy: 0.9976 - val_loss: 0.1179 - val_accuracy: 0.9565
# Epoch 10/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0114 - accuracy: 0.9976 - val_loss: 0.1128 - val_accuracy: 0.9565
# Epoch 11/150
# 41/41 [==============================] - 0s 2ms/step - loss: 0.0096 - accuracy: 0.9976 - val_loss: 0.1144 - val_accuracy: 0.9565
# Epoch 12/150
# 41/41 [==============================] - 0s 2ms/step - loss: 0.0103 - accuracy: 0.9976 - val_loss: 0.1109 - val_accuracy: 0.9565
# Epoch 13/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0089 - accuracy: 0.9976 - val_loss: 0.1080 - val_accuracy: 0.9565
# Epoch 14/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0102 - accuracy: 0.9951 - val_loss: 0.0996 - val_accuracy: 0.9565
# Epoch 15/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0098 - accuracy: 0.9976 - val_loss: 0.1010 - val_accuracy: 0.9565
# Epoch 16/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0077 - accuracy: 0.9976 - val_loss: 0.0996 - val_accuracy: 0.9565
# Epoch 17/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0088 - accuracy: 0.9976 - val_loss: 0.1019 - val_accuracy: 0.9565
# Epoch 18/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0070 - accuracy: 0.9951 - val_loss: 0.0969 - val_accuracy: 0.9565
# Epoch 19/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0073 - accuracy: 0.9976 - val_loss: 0.0948 - val_accuracy: 0.9565
# Epoch 20/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0065 - accuracy: 0.9976 - val_loss: 0.0953 - val_accuracy: 0.9565
# Epoch 21/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0053 - accuracy: 0.9976 - val_loss: 0.0926 - val_accuracy: 0.9565
# Epoch 22/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0062 - accuracy: 0.9976 - val_loss: 0.0921 - val_accuracy: 0.9565
# Epoch 23/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0055 - accuracy: 0.9976 - val_loss: 0.0892 - val_accuracy: 0.9565
# Epoch 24/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0043 - accuracy: 0.9976 - val_loss: 0.0875 - val_accuracy: 0.9565
# Epoch 25/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0042 - accuracy: 0.9976 - val_loss: 0.0872 - val_accuracy: 0.9565
# Epoch 26/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0047 - accuracy: 0.9976 - val_loss: 0.0879 - val_accuracy: 0.9565
# Epoch 27/150
# 41/41 [==============================] - 0s 2ms/step - loss: 0.0044 - accuracy: 0.9976 - val_loss: 0.0866 - val_accuracy: 0.9565
# Epoch 28/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0032 - accuracy: 1.0000 - val_loss: 0.0859 - val_accuracy: 0.9565
# Epoch 29/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0031 - accuracy: 1.0000 - val_loss: 0.0831 - val_accuracy: 0.9565
# Epoch 30/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0030 - accuracy: 1.0000 - val_loss: 0.0840 - val_accuracy: 0.9565
# Epoch 31/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0030 - accuracy: 1.0000 - val_loss: 0.0815 - val_accuracy: 0.9565
# Epoch 32/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.0811 - val_accuracy: 0.9565
# Epoch 33/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0034 - accuracy: 1.0000 - val_loss: 0.0834 - val_accuracy: 0.9565
# Epoch 34/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.0814 - val_accuracy: 0.9565
# Epoch 35/150
# 41/41 [==============================] - 0s 2ms/step - loss: 0.0025 - accuracy: 1.0000 - val_loss: 0.0798 - val_accuracy: 0.9565
# Epoch 36/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.0789 - val_accuracy: 0.9565
# Epoch 37/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0027 - accuracy: 1.0000 - val_loss: 0.0802 - val_accuracy: 0.9565
# Epoch 38/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0024 - accuracy: 1.0000 - val_loss: 0.0789 - val_accuracy: 0.9565
# Epoch 39/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0022 - accuracy: 1.0000 - val_loss: 0.0772 - val_accuracy: 0.9565
# Epoch 40/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0772 - val_accuracy: 0.9565
# Epoch 41/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0018 - accuracy: 1.0000 - val_loss: 0.0754 - val_accuracy: 0.9565
# Epoch 42/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0017 - accuracy: 1.0000 - val_loss: 0.0756 - val_accuracy: 0.9565
# Epoch 43/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.0747 - val_accuracy: 0.9565
# Epoch 44/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.0756 - val_accuracy: 0.9565
# Epoch 45/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0015 - accuracy: 1.0000 - val_loss: 0.0748 - val_accuracy: 0.9565
# Epoch 46/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0738 - val_accuracy: 0.9565
# Epoch 47/150
# 41/41 [==============================] - 0s 4ms/step - loss: 0.0014 - accuracy: 1.0000 - val_loss: 0.0739 - val_accuracy: 0.9565
# Epoch 48/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0013 - accuracy: 1.0000 - val_loss: 0.0732 - val_accuracy: 0.9565
# Epoch 49/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0738 - val_accuracy: 0.9565
# Epoch 50/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 51/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0732 - val_accuracy: 0.9565
# Epoch 52/150
# 41/41 [==============================] - 0s 2ms/step - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 53/150
# 41/41 [==============================] - 0s 2ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0740 - val_accuracy: 0.9565
# Epoch 54/150
# 41/41 [==============================] - 0s 3ms/step - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0728 - val_accuracy: 0.9565
# Epoch 55/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.4609e-04 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 56/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.3717e-04 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 57/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.4558e-04 - accuracy: 1.0000 - val_loss: 0.0728 - val_accuracy: 0.9565
# Epoch 58/150
# 41/41 [==============================] - 0s 3ms/step - loss: 8.6047e-04 - accuracy: 1.0000 - val_loss: 0.0728 - val_accuracy: 0.9565
# Epoch 59/150
# 41/41 [==============================] - 0s 3ms/step - loss: 8.0544e-04 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 60/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.7938e-04 - accuracy: 1.0000 - val_loss: 0.0721 - val_accuracy: 0.9565
# Epoch 61/150
# 41/41 [==============================] - 0s 3ms/step - loss: 8.2427e-04 - accuracy: 1.0000 - val_loss: 0.0730 - val_accuracy: 0.9565
# Epoch 62/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.5896e-04 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 63/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.3733e-04 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 64/150
# 41/41 [==============================] - 0s 3ms/step - loss: 6.6164e-04 - accuracy: 1.0000 - val_loss: 0.0730 - val_accuracy: 0.9565
# Epoch 65/150
# 41/41 [==============================] - 0s 3ms/step - loss: 6.7486e-04 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 66/150
# 41/41 [==============================] - 0s 3ms/step - loss: 6.5500e-04 - accuracy: 1.0000 - val_loss: 0.0730 - val_accuracy: 0.9565
# Epoch 67/150
# 41/41 [==============================] - 0s 3ms/step - loss: 5.9986e-04 - accuracy: 1.0000 - val_loss: 0.0730 - val_accuracy: 0.9565
# Epoch 68/150
# 41/41 [==============================] - 0s 2ms/step - loss: 6.0424e-04 - accuracy: 1.0000 - val_loss: 0.0725 - val_accuracy: 0.9565
# Epoch 69/150
# 41/41 [==============================] - 0s 2ms/step - loss: 5.8697e-04 - accuracy: 1.0000 - val_loss: 0.0729 - val_accuracy: 0.9565
# Epoch 70/150
# 41/41 [==============================] - 0s 3ms/step - loss: 5.5504e-04 - accuracy: 1.0000 - val_loss: 0.0732 - val_accuracy: 0.9565
# Epoch 71/150
# 41/41 [==============================] - 0s 3ms/step - loss: 5.2555e-04 - accuracy: 1.0000 - val_loss: 0.0733 - val_accuracy: 0.9565
# Epoch 72/150
# 41/41 [==============================] - 0s 3ms/step - loss: 5.3449e-04 - accuracy: 1.0000 - val_loss: 0.0737 - val_accuracy: 0.9565
# Epoch 73/150
# 41/41 [==============================] - 0s 3ms/step - loss: 5.7659e-04 - accuracy: 1.0000 - val_loss: 0.0733 - val_accuracy: 0.9565
# Epoch 74/150
# 41/41 [==============================] - 0s 3ms/step - loss: 4.9512e-04 - accuracy: 1.0000 - val_loss: 0.0736 - val_accuracy: 0.9565
# Epoch 75/150
# 41/41 [==============================] - 0s 3ms/step - loss: 4.8333e-04 - accuracy: 1.0000 - val_loss: 0.0738 - val_accuracy: 0.9565
# Epoch 76/150
# 41/41 [==============================] - 0s 3ms/step - loss: 4.5753e-04 - accuracy: 1.0000 - val_loss: 0.0735 - val_accuracy: 0.9565
# Epoch 77/150
# 41/41 [==============================] - 0s 3ms/step - loss: 4.5510e-04 - accuracy: 1.0000 - val_loss: 0.0735 - val_accuracy: 0.9565
# Epoch 78/150
# 41/41 [==============================] - 0s 3ms/step - loss: 4.2470e-04 - accuracy: 1.0000 - val_loss: 0.0734 - val_accuracy: 0.9565
# Epoch 79/150
# 41/41 [==============================] - 0s 3ms/step - loss: 4.1901e-04 - accuracy: 1.0000 - val_loss: 0.0736 - val_accuracy: 0.9565
# Epoch 80/150
# 41/41 [==============================] - 0s 3ms/step - loss: 4.3531e-04 - accuracy: 1.0000 - val_loss: 0.0739 - val_accuracy: 0.9565
# Epoch 81/150
# 41/41 [==============================] - 0s 3ms/step - loss: 4.1395e-04 - accuracy: 1.0000 - val_loss: 0.0741 - val_accuracy: 0.9565
# Epoch 82/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.8604e-04 - accuracy: 1.0000 - val_loss: 0.0737 - val_accuracy: 0.9565
# Epoch 83/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.8929e-04 - accuracy: 1.0000 - val_loss: 0.0736 - val_accuracy: 0.9565
# Epoch 84/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.6475e-04 - accuracy: 1.0000 - val_loss: 0.0738 - val_accuracy: 0.9565
# Epoch 85/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.4930e-04 - accuracy: 1.0000 - val_loss: 0.0748 - val_accuracy: 0.9565
# Epoch 86/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.4698e-04 - accuracy: 1.0000 - val_loss: 0.0741 - val_accuracy: 0.9565
# Epoch 87/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.2428e-04 - accuracy: 1.0000 - val_loss: 0.0743 - val_accuracy: 0.9565
# Epoch 88/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.2911e-04 - accuracy: 1.0000 - val_loss: 0.0741 - val_accuracy: 0.9565
# Epoch 89/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.0971e-04 - accuracy: 1.0000 - val_loss: 0.0746 - val_accuracy: 0.9565
# Epoch 90/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.0122e-04 - accuracy: 1.0000 - val_loss: 0.0743 - val_accuracy: 0.9565
# Epoch 91/150
# 41/41 [==============================] - 0s 3ms/step - loss: 3.0233e-04 - accuracy: 1.0000 - val_loss: 0.0748 - val_accuracy: 0.9565
# Epoch 92/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.8334e-04 - accuracy: 1.0000 - val_loss: 0.0747 - val_accuracy: 0.9565
# Epoch 93/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.8162e-04 - accuracy: 1.0000 - val_loss: 0.0744 - val_accuracy: 0.9565
# Epoch 94/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.6339e-04 - accuracy: 1.0000 - val_loss: 0.0745 - val_accuracy: 0.9565
# Epoch 95/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.6572e-04 - accuracy: 1.0000 - val_loss: 0.0743 - val_accuracy: 0.9565
# Epoch 96/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.5937e-04 - accuracy: 1.0000 - val_loss: 0.0742 - val_accuracy: 0.9565
# Epoch 97/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.5951e-04 - accuracy: 1.0000 - val_loss: 0.0744 - val_accuracy: 0.9565
# Epoch 98/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.4590e-04 - accuracy: 1.0000 - val_loss: 0.0746 - val_accuracy: 0.9565
# Epoch 99/150
# 41/41 [==============================] - 0s 2ms/step - loss: 2.3904e-04 - accuracy: 1.0000 - val_loss: 0.0743 - val_accuracy: 0.9565
# Epoch 100/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.2915e-04 - accuracy: 1.0000 - val_loss: 0.0742 - val_accuracy: 0.9565
# Epoch 101/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.3068e-04 - accuracy: 1.0000 - val_loss: 0.0744 - val_accuracy: 0.9565
# Epoch 102/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.3191e-04 - accuracy: 1.0000 - val_loss: 0.0745 - val_accuracy: 0.9565
# Epoch 103/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.1442e-04 - accuracy: 1.0000 - val_loss: 0.0747 - val_accuracy: 0.9565
# Epoch 104/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.0746e-04 - accuracy: 1.0000 - val_loss: 0.0745 - val_accuracy: 0.9565
# Epoch 105/150
# 41/41 [==============================] - 0s 3ms/step - loss: 2.0825e-04 - accuracy: 1.0000 - val_loss: 0.0746 - val_accuracy: 0.9565
# Epoch 106/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.9550e-04 - accuracy: 1.0000 - val_loss: 0.0746 - val_accuracy: 0.9565
# Epoch 107/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.9435e-04 - accuracy: 1.0000 - val_loss: 0.0744 - val_accuracy: 0.9565
# Epoch 108/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.9360e-04 - accuracy: 1.0000 - val_loss: 0.0745 - val_accuracy: 0.9565
# Epoch 109/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.8472e-04 - accuracy: 1.0000 - val_loss: 0.0741 - val_accuracy: 0.9565
# Epoch 110/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.8217e-04 - accuracy: 1.0000 - val_loss: 0.0740 - val_accuracy: 0.9783
# Epoch 111/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.7291e-04 - accuracy: 1.0000 - val_loss: 0.0748 - val_accuracy: 0.9565
# Epoch 112/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.6940e-04 - accuracy: 1.0000 - val_loss: 0.0744 - val_accuracy: 0.9565
# Epoch 113/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.6555e-04 - accuracy: 1.0000 - val_loss: 0.0747 - val_accuracy: 0.9565
# Epoch 114/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.6032e-04 - accuracy: 1.0000 - val_loss: 0.0748 - val_accuracy: 0.9565
# Epoch 115/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.5528e-04 - accuracy: 1.0000 - val_loss: 0.0753 - val_accuracy: 0.9565
# Epoch 116/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.5350e-04 - accuracy: 1.0000 - val_loss: 0.0749 - val_accuracy: 0.9565
# Epoch 117/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.4896e-04 - accuracy: 1.0000 - val_loss: 0.0751 - val_accuracy: 0.9565
# Epoch 118/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.4650e-04 - accuracy: 1.0000 - val_loss: 0.0748 - val_accuracy: 0.9565
# Epoch 119/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.4358e-04 - accuracy: 1.0000 - val_loss: 0.0748 - val_accuracy: 0.9565
# Epoch 120/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.4008e-04 - accuracy: 1.0000 - val_loss: 0.0746 - val_accuracy: 0.9783
# Epoch 121/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.3766e-04 - accuracy: 1.0000 - val_loss: 0.0750 - val_accuracy: 0.9565
# Epoch 122/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.3563e-04 - accuracy: 1.0000 - val_loss: 0.0750 - val_accuracy: 0.9565
# Epoch 123/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.3052e-04 - accuracy: 1.0000 - val_loss: 0.0752 - val_accuracy: 0.9783
# Epoch 124/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.2641e-04 - accuracy: 1.0000 - val_loss: 0.0752 - val_accuracy: 0.9565
# Epoch 125/150
# 41/41 [==============================] - 0s 4ms/step - loss: 1.2332e-04 - accuracy: 1.0000 - val_loss: 0.0753 - val_accuracy: 0.9783
# Epoch 126/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.2321e-04 - accuracy: 1.0000 - val_loss: 0.0749 - val_accuracy: 0.9783
# Epoch 127/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.1830e-04 - accuracy: 1.0000 - val_loss: 0.0753 - val_accuracy: 0.9783
# Epoch 128/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.1651e-04 - accuracy: 1.0000 - val_loss: 0.0747 - val_accuracy: 0.9783
# Epoch 129/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.1702e-04 - accuracy: 1.0000 - val_loss: 0.0756 - val_accuracy: 0.9783
# Epoch 130/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.1082e-04 - accuracy: 1.0000 - val_loss: 0.0756 - val_accuracy: 0.9565
# Epoch 131/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.0683e-04 - accuracy: 1.0000 - val_loss: 0.0755 - val_accuracy: 0.9783
# Epoch 132/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.0565e-04 - accuracy: 1.0000 - val_loss: 0.0754 - val_accuracy: 0.9783
# Epoch 133/150
# 41/41 [==============================] - 0s 3ms/step - loss: 1.0226e-04 - accuracy: 1.0000 - val_loss: 0.0759 - val_accuracy: 0.9783
# Epoch 134/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.9745e-05 - accuracy: 1.0000 - val_loss: 0.0757 - val_accuracy: 0.9783
# Epoch 135/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.8098e-05 - accuracy: 1.0000 - val_loss: 0.0754 - val_accuracy: 0.9783
# Epoch 136/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.4453e-05 - accuracy: 1.0000 - val_loss: 0.0752 - val_accuracy: 0.9783
# Epoch 137/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.3862e-05 - accuracy: 1.0000 - val_loss: 0.0756 - val_accuracy: 0.9783
# Epoch 138/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.1716e-05 - accuracy: 1.0000 - val_loss: 0.0755 - val_accuracy: 0.9783
# Epoch 139/150
# 41/41 [==============================] - 0s 3ms/step - loss: 9.0364e-05 - accuracy: 1.0000 - val_loss: 0.0757 - val_accuracy: 0.9783
# Epoch 140/150
# 41/41 [==============================] - 0s 3ms/step - loss: 8.7444e-05 - accuracy: 1.0000 - val_loss: 0.0757 - val_accuracy: 0.9783
# Epoch 141/150
# 41/41 [==============================] - 0s 3ms/step - loss: 8.5096e-05 - accuracy: 1.0000 - val_loss: 0.0756 - val_accuracy: 0.9783
# Epoch 142/150
# 41/41 [==============================] - 0s 3ms/step - loss: 8.4241e-05 - accuracy: 1.0000 - val_loss: 0.0756 - val_accuracy: 0.9783
# Epoch 143/150
# 41/41 [==============================] - 0s 3ms/step - loss: 8.1981e-05 - accuracy: 1.0000 - val_loss: 0.0758 - val_accuracy: 0.9783
# Epoch 144/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.9828e-05 - accuracy: 1.0000 - val_loss: 0.0753 - val_accuracy: 0.9783
# Epoch 145/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.7301e-05 - accuracy: 1.0000 - val_loss: 0.0755 - val_accuracy: 0.9783
# Epoch 146/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.4824e-05 - accuracy: 1.0000 - val_loss: 0.0756 - val_accuracy: 0.9783
# Epoch 147/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.4779e-05 - accuracy: 1.0000 - val_loss: 0.0758 - val_accuracy: 0.9783
# Epoch 148/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.2513e-05 - accuracy: 1.0000 - val_loss: 0.0758 - val_accuracy: 0.9783
# Epoch 149/150
# 41/41 [==============================] - 0s 3ms/step - loss: 7.0977e-05 - accuracy: 1.0000 - val_loss: 0.0755 - val_accuracy: 0.9783
# Epoch 150/150
# 41/41 [==============================] - 0s 3ms/step - loss: 6.9652e-05 - accuracy: 1.0000 - val_loss: 0.0760 - val_accuracy: 0.9783

# %%
# see the acuracy of the model
eval_model=classifier.evaluate(x_train.T, y_train.T)
print("Accuracy of a model", eval_model)
# 15/15 [==============================] - 0s 1ms/step - loss: 0.0077 - accuracy: 0.9978
# Accuracy of a model [0.0077413348481059074, 0.997802197933197]

# %%
# predict the outputs of unseen dataset
fromAnnYPred = classifier.predict(x_test.T)
fromAnnYPred = (fromAnnYPred > 0.5)

# %%
# create a confusion matrix of unseen dataset
cm = confusion_matrix(y_test, fromAnnYPred)
print(cm)
# show the accuracy score of unseen dataset
print(accuracy_score(y_test, fromAnnYPred))
# 4/4 [==============================] - 0s 3ms/step
# [[69  2]
#  [ 1 42]]
# 0.9736842105263158

# %%
history_dict = resultFit.history
print(history_dict.keys())
dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


# %%
# Plot accuracy of a model with recpect to number of epochs.
plt.plot(resultFit.history['accuracy'])
plt.plot(resultFit.history['val_accuracy'])
plt.title('Accuracy of a model')
plt.ylabel('Accuracy percent')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%
# Plot loss of a model with respect to number of epochs
plt.plot(resultFit.history['loss'])
plt.plot(resultFit.history['val_loss'])
plt.title('Loss of a model')
plt.ylabel('Loss function')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
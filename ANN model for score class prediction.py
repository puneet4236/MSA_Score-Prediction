
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_excel('test_train_data_of features.xlsx')
X = dataset.iloc[:, 0:9].values
y = dataset.iloc[:, 9].values



from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()


classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))

classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax')) 

classifier.compile(optimizer = 'SGD', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 1, epochs = 1000)


y_pred = classifier.predict(X_test)






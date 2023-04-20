import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from joblib import dump, load
import scipy.io
import xgboost as xgb

print('\n\n#------------------------------------------------------------\n')
mat_data = scipy.io.loadmat('MNIST.mat')
print('keys -> ', mat_data.keys())

X = mat_data['train_fea']
Y = mat_data['train_gnd']
Y = Y - 1

test_X = mat_data['test_fea']
test_Y = mat_data['test_gnd']
test_Y -= 1

print('train data shape -> ', X.shape)
print('test data shape -> ', test_X.shape)
print('train labels shape -> ', Y.shape)
print('test labels shape -> ', test_Y.shape)

clf = xgb.XGBClassifier()
clf.fit(X, Y)

# Evaluate the performance of the classifier on the testing data
y_pred = clf.predict(test_X)
accuracy = accuracy_score(test_Y, y_pred)
print(f"Accuracy: {accuracy}")

print('\n#------------------------------------------------------------\n\n')
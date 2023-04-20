import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Plot the first 10 images from the test set
shift = 0
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    if i < 10:
        ax.imshow(test_X[i+shift].reshape(28, 28), cmap='gray')
        ax.set_xlabel(f"Label: {test_Y[i+shift]+1}, pred: {y_pred[i+shift]+1}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# plot the confusion matrix
cm = confusion_matrix(test_Y, y_pred)
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=np.arange(10),
       yticklabels=np.arange(10),
       xlabel='Predicted label',
       ylabel='True label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
fmt = '.0f' 
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.show()

print('\n#------------------------------------------------------------\n\n')
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from joblib import dump, load


# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32)
y = mnist.target.astype(np.int32)



# Subset the data to the first 1000 images for training and the second 1000 for testing
X_train = X[:1000]
y_train = y[:1000]
X_test = X[1000:2000]
y_test = y[1000:2000]

# Create a logistic regression model and train it
model = LogisticRegression(max_iter=100)


model.fit(X_train, y_train)
# model = load('my_model.joblib')

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Save the model to a file
dump(model, 'my_model.joblib')

y_pred = model.predict(X_test[:10])
print("Predictions:")
print(y_pred)
print("actual:")
print(y_test[:10])

print(X_test.shape)
# Reshape the images to 8x8 arrays
# Extract the first five images from the DataFrame
# Reshape each row of X_test to a 28x28 image
images = X_test.values
images = images.reshape(-1, 28, 28)

# Plot the first five images in a 2x3 grid
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(6, 4))
for i, ax in enumerate(axes.flat):
    if i < len(images):
        ax.imshow(images[i], cmap='gray')
        s = str(y_test[i+1000]) + "|" + str(y_pred[i])
        ax.set_title(s)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
# K-Nearest Neighbors (KNN) Classifier Implementation

This repository contains an implementation of the K-Nearest Neighbors (KNN) classifier using Python. The KNN algorithm is implemented in the `KNN` class with methods for fitting the model (`fit`), making predictions (`predict`), and calculating Euclidean distance (`ecd_distance`).

## Overview

- [Introduction](#introduction)
- [Usage](#usage)
- [File Description](#file-description)
- [Example Usage](#example-usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

K-Nearest Neighbors (KNN) is a simple and effective machine learning algorithm used for classification and regression tasks. The KNN algorithm classifies a data point based on the majority class of its k-nearest neighbors. This implementation provides a basic version of the KNN algorithm in Python, utilizing the Euclidean distance metric to calculate the proximity between data points.

## Usage

To use this code:

1. Clone the repository:

    ```
    git clone https://github.com/Mukund604/KNN-Classifier.git
    ```

2. Ensure you have Python installed along with the necessary libraries, particularly numpy and collections.

3. Import the `KNN` class from `knn_classifier.py` into your Python script or Jupyter Notebook.

4. Create an instance of the `KNN` class and use the `fit` method to train the model with training data.

5. Utilize the `predict` method to make predictions on new data points based on their nearest neighbors.

## File Description

- `knn_classifier.py`: Python script containing the implementation of the KNN classifier.

## Example Usage

```python
# Example usage of the KNN classifier

import numpy as np
from knn_classifier import KNN

# Example data
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([0, 1, 0])

# Create an instance of the KNN classifier
knn = KNN(k=3)

# Fit the model with training data
knn.fit(X_train, y_train)

# Example data for prediction
X_test = np.array([[2, 3], [4, 5]])

# Make predictions on new data points
predictions = knn.predict(X_test)
print("Predictions:", predictions)

#Importing the required Libraries
import numpy as np
from KNN import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split


#Using the inbuilt Breast Cancer dataset from scikit-learn library
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

#Splitting the dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#Initializing the Classifier and fitting the model with the given dataset
clf = KNN(k = 4)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


#Checking the Accuracy of the model
def Accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print("The Accuracy of the model is:",Accuracy(y_test, y_pred))


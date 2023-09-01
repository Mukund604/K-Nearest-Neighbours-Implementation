#Importing the Libraies 
import numpy as np
from collections import Counter


#Our function to calculate the distance bewteen the points.
def ecd_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))



class KNN:
    #Making a constructor function for our classifier
    def __init__(self,k = 3):
        self.k = k
        
    #Fit method
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
        
    #Making Predicitons
    def predict(self,X):
        pred = [self._predict(x) for x in X]
        return np.array(pred)
        
    def _predict(self,x):
        distances = [ecd_distance(x,x_train) for x_train in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:16:32 2019

@author: y-din
"""

# Prepare the dataset
from six.moves import urllib

print("Could not download MNIST data from mldata.org, trying alternative...")

# Alternative method to load MNIST, if mldata.org is down
from scipy.io import loadmat
mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"
response = urllib.request.urlopen(mnist_alternative_url)
with open(mnist_path, "wb") as f:
    content = response.read()
    f.write(content)
mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}
print("Success!")

X,y = mnist['data'], mnist['target']

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#==============================================================================
# Set up environment
#==============================================================================
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load scikit's support vector machine library
from sklearn.svm import SVC

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)

# Load scikit's grid search cross validation library
from sklearn.model_selection import GridSearchCV

# Import accuracy score library
from sklearn.metrics import accuracy_score

# Set grid search parameters
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [100, 300, 784]}]
rfc = RandomForestClassifier()

#==============================================================================
# Perform a 3-fold grid search and train the Random Forest Classifier
#==============================================================================
# Train rfc model across 3 folds
grid_search = GridSearchCV(rfc, param_grid, cv = 3)
grid_search.fit(X_train, y_train)

# Review best model
grid_search.best_estimator_

# Test final model and calculate accuracy
final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)
print("Accuracy of the ramdom forest classifier is : ", accuracy_score(y_test, y_pred))

#==============================================================================
# Perform a 3-fold grid search and train the Support Vector Machine Classifier
#==============================================================================
# Normalize the training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
std_X_train = scaler.transform(X_train)
std_X_test = scaler.transform(X_test)

# Set grid parameters
param_grid2 = [
        {'kernel': ['linear'], 'C': [10., 30., 100.],'probability':[False]},
        {'kernel': ['poly'], 'C': [1.0, 3.0, 5.0],'gamma': [0.01, 0.03, 0.08],'probability':[False]},
        {'kernel': ['rbf'],'C': [2.8],'gamma': [0.0073],'probability': [False]}
]
svm = SVC()

# Train svc model across 3 folds using the first 10000 datapoints
grid_search = GridSearchCV(svm, param_grid2, cv = 3)
grid_search.fit(std_X_train[:10000], y_train[:10000])

# Review the best estimators from grid search
grid_search.best_estimator_

# Train selected model with the remaining 50000 data points.
svm_trained = SVC(C=1.0, gamma = 0.01, kernel='poly', probability=False)
svm_trained.fit(std_X_train[10000:], y_train[10000:])

# Test trained model and calculate accuracy
y_pred2 = svm_trained.predict(X_test)
print(\"Accuracy of the support vector classifier is: \", svm_trained.score(std_X_test, y_test))
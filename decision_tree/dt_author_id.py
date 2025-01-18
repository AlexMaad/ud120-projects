#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess

# Preprocess the dataset
features_train, features_test, labels_train, labels_test = preprocess()

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier(min_samples_split=40)  # Set a parameter for the tree
t0 = time()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time() - t0, 3)} seconds")

# Make predictions on the test set
t1 = time()
pred = clf.predict(features_test)
print(f"Prediction time: {round(time() - t1, 3)} seconds")

# Calculate and print the accuracy of the classifier
accuracy = accuracy_score(labels_test, pred)
print(f"Accuracy: {accuracy}")

# Optional: Explore feature importance or tree structure
print(f"Number of features: {len(features_train[0])}")




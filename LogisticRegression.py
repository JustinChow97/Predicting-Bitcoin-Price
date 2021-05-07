#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:34:43 2021

@author: justinchow
"""

# Import libaries and packages
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support as score

# Import Bitcoin price data
# data = pd.read_excel('old Bitcoin Data.xlsx')
data = pd.read_excel("FINAL_BITCOIN_DATA.xlsx", engine="openpyxl")


# Set X as feature data
# Set y as label data

X = data.drop(["Pos or Neg Change in Close Price"], axis = 1)

y = data["Pos or Neg Change in Close Price"]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# =============================================================================
# Create and fit training data to model
# increase the max iterations to avoid convergence error
logreg = LogisticRegression(solver='lbfgs', max_iter=3000).fit(X_train,y_train)

# Print Results
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
# =============================================================================



predicted = logreg.predict(X_test)
fscore = f1_score(y_test, predicted, average = 'weighted')
precision = precision_score(y_test, predicted, average='weighted')
recall = recall_score(y_test, predicted, average='weighted')

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))

# print('support: {}'.format(support))

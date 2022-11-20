import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')


X = df['Glucose'].values.reshape(-1, 1)
y = df['Age'].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.2,random_state = 1)

X_train3, X_test3, y_train3, y_test = train_test_split(X, y, test_size = 0.2,random_state = 2)


import pytest

def test_B():
    assert np.array_equal(X_train,X_train2)

def test_C():
    assert np.array_equal(X_train,X_train3)


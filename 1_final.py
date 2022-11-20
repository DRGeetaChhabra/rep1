import numpy as np
import pandas as pd

df = pd.read_csv('diabetes.csv')


X = df['Glucose'].values.reshape(-1, 1)
y = df['Age'].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 1)

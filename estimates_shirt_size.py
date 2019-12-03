import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("./data/processed_data.csv", index_col=0)

data_clean = data.drop(columns=['key'])
data_clean.fillna(0, inplace=True)

X = data_clean.drop(columns=['estimate'])
y = data_clean['estimate']

bins_labels = ['S+M', 'L+XL']
bins = [0, y.max()/2, y.max()+50] 

y_binned = pd.cut(y, bins=bins, labels=bins_labels)

print("my bins are: ", bins)

# Random Forest - with pred tests
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestClassifier(n_estimators=20)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

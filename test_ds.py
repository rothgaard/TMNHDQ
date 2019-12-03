import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("./data/processed_data.csv", index_col=0)

data_clean = data.drop(columns=['key'])
data_clean.fillna(0, inplace=True)

# get correlations of each features in dataset
corrmat = data_clean.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
sns.set(font_scale=1.2)
g=sns.heatmap(data_clean[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# plot heatmap
plt.yticks(rotation=0)
plt.savefig('corr_heatmap.png')


from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

# Lasso 
X = data_clean.drop(columns=['estimate'])
y = data_clean['estimate']

# Save column names first
names = X.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_X = scaler.fit_transform(X)
X = pd.DataFrame(scaled_X, columns=names)

reg = LassoCV(max_iter=5000)
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + 
    " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

plt.figure(2)
imp_coef = coef.sort_values()
plt.figure(figsize=(12,12))
imp_coef.plot(kind = "barh")
plt.title("Feature importance using LassoCV Model")

plt.savefig('LassoCV.png')

# Random Forest - Understand Key Factors
rf = RandomForestRegressor(n_estimators=200)
rf.fit(X, y)

print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True))

# Random Forest - with pred tests
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn import metrics
print('Mean of estimate:', y.mean())
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

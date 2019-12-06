import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import RFECV

data = pd.read_csv("./processed_data.csv", index_col=0)
target = pd.read_csv("./processed_target_data.csv", index_col=0)

data_clean = data.drop(columns=['doc_key'])
data_clean.fillna(0, inplace=True)

target_clean = target.drop(columns=['doc_key'])
target_clean.fillna(0, inplace=True)


# Data Extraction
X = data_clean.drop(columns=['estimate'])
y = data_clean['estimate']

import numpy as np

# # LASSO me baby

# # Save column names first
# names = X.columns
# # Create the Scaler object
# scaler = preprocessing.StandardScaler()
# # Fit your data on the scaler object
# scaled_X = scaler.fit_transform(X)
# X_Lasso = pd.DataFrame(scaled_X, columns=names)
#
# reg = LassoCV(max_iter=5000)
# reg.fit(X_Lasso, y)
# print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
# print("Best score using built-in LassoCV: %f" % reg.score(X_Lasso,y))
#
# coef = pd.Series(reg.coef_, index = names)
# features_to_remove = [item for item in names if coef[item] == 0]
#
# # imp_coef = coef.sort_values()
# # plt.figure(figsize=(12,12))
# # imp_coef.plot(kind = "barh")
# # plt.title("Feature importance using LassoCV Model")
# # plt.savefig('LassoCV.png')
# features_kept = [item for item in names if coef[item] != 0]
#
# #print("Removed the following features: ", features_to_remove)
# print("Features Kept: ", features_kept)
#
# X = X.drop(columns=features_to_remove)
# target_clean = target_clean.drop(columns=features_to_remove)

# End Lasso
bin_labels_5 = ['XS', 'S', 'M', 'L', 'XL']
bin_labels_2 = ['S', 'L']
bin_labels_3 = ['S', 'M', 'L']

bin_labels = bin_labels_2
y_bins = pd.qcut(y, q=len(bin_labels), labels=bin_labels)

from sklearn.feature_extraction.text import TfidfTransformer

# # TF-IDF
# tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
# tfidf_transformer = tfidf_transformer.fit_transform(X)
# X = pd.DataFrame(tfidf_transformer.toarray(), columns=X.columns)

model = MultinomialNB()
selector = RFECV(estimator=model, min_features_to_select=5, step=1, cv=5)
selector = selector.fit(X, y_bins)

cv_results = cross_val_score(model, X, y_bins, cv=5, scoring='accuracy')

print("=== Results with Multinomial Naive Bayes ===")
print('Bins: ', bin_labels)
print('Counts:')
print(y_bins.value_counts())
print('\nCurrent accuracy of model (via cross Validation) is:', cv_results.mean() * 100)

print("=== Prediction of target ===")
prediction = selector.predict(target_clean[0:1])
print(target["doc_key"].to_numpy() , prediction)
print("Range(days): [{},{}]".format(y.quantile(q=bin_labels.index(prediction)/len(bin_labels)),
      y.quantile(q=(bin_labels.index(prediction)+1)/len(bin_labels))))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv("./data/processed_data.csv", index_col=0)

data_clean = data.drop(columns=['key'])
data_clean.fillna(0, inplace=True)

# get correlations of each features in dataset
corrmat = data_clean.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
sns.set(font_scale=1.2)
g=sns.heatmap(data_clean[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#print(data_clean.to_string())
plt.yticks(rotation=0)
plt.savefig('corr_heatmap.png')
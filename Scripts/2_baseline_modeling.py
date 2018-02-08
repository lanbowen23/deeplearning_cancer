import pandas as pd
import numpy as np
import importlib
from Scripts import baseline_models
importlib.reload(baseline_models)
from sklearn import preprocessing
import matplotlib.pylab as plt
from sklearn.decomposition import PCA


df_all_data = pd.read_csv("~/ghub/deeplearning_cancer/Data/all_data_combined.csv")
df_all_data.head(10)

# check correlation
baseline_models.correlation_info(df_all_data.loc[:,'Leukemia_':'Liver_car'],0.7)
#Correlation filter > 0.7 :  1  features from the dataset

# naive bayes model 3 fold cv
baseline_models.naivebayes(df_all_data.loc[:,'Leukemia_':'class'],3)
#Average naivebayes accuracy is: 0.50395133322

## pca analysis
x_data =df_all_data.loc[:,'Leukemia_':'Liver_car']
x_scaled = pd.DataFrame(preprocessing.scale(x_data))
x_scaled.columns = x_data.columns
#plt.scatter(x_scaled.iloc[:,2],x_scaled.iloc[:,1])
#x_scaled.iloc[:,0:5].plot()
pca = PCA(n_components=2)
print (pca.fit(x_scaled).explained_variance_)
pca.fit(x_scaled).explained_variance_ratio_

principalComponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
finalDf = pd.concat([principalDf, df_all_data['class']], axis = 1)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf.loc[:,'class'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color)
ax.legend(targets)
ax.grid()

corr_matrix = x_scaled.corr()
eig_val_cov, eig_vec_cov = np.linalg.eig(corr_matrix)


## logistic regression
baseline_models.logregression(df_all_data.loc[:,'Leukemia_':'class'],2)
# Logistic regression score 0.566191938143


## grid search for lasso and xgboost
scoremat = baseline_models.grid_search('LRLASSO',df_all_data.loc[:,'Leukemia_':'class'],2,[0.001,0.01,0.1,1.0],True)
#scoremat array([ 0.52264199,  0.56396043,  0.56559497,  0.56660413])


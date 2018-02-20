import pandas as pd
import numpy as np
import importlib
from Scripts import baseline_models
importlib.reload(baseline_models)
from sklearn import preprocessing
import matplotlib.pylab as plt
from sklearn.decomposition import PCA

# ## test dataset from uci
# df_all_data = pd.read_csv("~/ghub/deeplearning_cancer/Data/breast_cancer_uci.csv")
# df_all_data.head(10)


## experimental cancer data
df_all_data = pd.read_csv("~/ghub/deeplearning_cancer/Data/all_data_combined.csv")
df_all_data = df_all_data.iloc[:,1:]
df_all_data.head(10)
print (df_all_data.shape)
### center data
x_data =df_all_data.iloc[:,0:df_all_data.shape[1]-1]
x_scaled = pd.DataFrame(preprocessing.scale(x_data))
x_scaled.columns = x_data.columns
df_all_data.iloc[:,0:df_all_data.shape[1]-1] = x_scaled
df_all_data.head(10)

## visualize data
df_all_data.iloc[:,1:5].diff().hist(color='k', alpha=0.5, bins=50)
df_all_data.iloc[:,1:10].plot.box()



# check correlation
todrop = baseline_models.correlation_info(df_all_data.iloc[:,0:df_all_data.shape[1]],0.9,drop=1,draw=0)
df_all_data.drop(todrop, axis=1, inplace=True)
print (df_all_data.shape)
#Correlation filter > 0.7 :  1  features from the dataset


## pca analysis
#plt.scatter(x_scaled.iloc[:,2],x_scaled.iloc[:,1])
#x_scaled.iloc[:,0:5].plot()
pca = PCA(n_components=30)
print (pca.fit(x_scaled).explained_variance_)
plt.plot(pca.fit(x_scaled).explained_variance_ratio_)

principalComponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
finalDf = pd.concat([principalDf, df_all_data['readmit']], axis = 1)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf.loc[:,'readmit'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color)
ax.legend(targets)
ax.grid()

corr_matrix = x_scaled.corr()
eig_val_cov, eig_vec_cov = np.linalg.eig(corr_matrix)


# naive bayes model 3 fold cv
baseline_models.naivebayes(df_all_data.iloc[:,0:df_all_data.shape[1]],3)
#Average naivebayes accuracy is: 0.50395133322

## logistic regression
baseline_models.logregression(df_all_data.iloc[:,0:df_all_data.shape[1]],3)
# Logistic regression score 0.822026022305


## svm regression
baseline_models.lsvm(df_all_data.iloc[:,0:df_all_data.shape[1]],3)
#svm score 0.811802973978

## grid search for lasso and xgboost
scoremat = baseline_models.grid_search('LRLASSO',df_all_data.iloc[:,0:df_all_data.shape[1]],2,[0.01,0.1,1.0],True)
print(scoremat)
#[ 0.5         0.75511152  0.81923792  0.83410781]


scoremat = baseline_models.grid_search('LSVM',df_all_data.iloc[:,0:df_all_data.shape[1]],2,[0.01,0.1,1.0,10.0],True)
print(scoremat)
##array[ 0.82574349  0.82713755  0.83271375  0.83410781]






# # analyze effect of regularization on linear model
# from sklearn.linear_model import LogisticRegression
# LogReg = LogisticRegression()
# LogReg.fit(df_all_data.iloc[:,0:df_all_data.shape[1]-1], df_all_data.iloc[:,df_all_data.shape[1]-1])
# plt.plot(LogReg.coef_[0])
# LogReg2 = LogisticRegression(C=.001, penalty='l1', tol=1e-6)
# LogReg2.fit(df_all_data.iloc[:,0:df_all_data.shape[1]-1], df_all_data.iloc[:,df_all_data.shape[1]-1])
# plt.plot(LogReg2.coef_[0])
# plt.scatter(LogReg.coef_[0],LogReg2.coef_[0])
# np.count_nonzero(LogReg2.coef_[0])
# np.count_nonzero(LogReg.coef_[0])
# #
# #
# import statsmodels.api as sm
# logit = sm.Logit(df_all_data.iloc[:,df_all_data.shape[1]-1],df_all_data.iloc[:,0:df_all_data.shape[1]-1])
# result = logit.fit()
# result.summary()

result={}
result["nbayes"]=0.71
result["logreg"]=0.84
result["svm"]=0.82
result["lasso:.01"]=0.75
result["lasso:.1"]=0.81
result["lasso:1.0"]=0.84

plt.ylim(0,1)
plt.bar(result.keys(), result.values(),align='center', alpha=0.5, color='g')
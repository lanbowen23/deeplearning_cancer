import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import numpy as np

def correlation_info(datamatrix,th):
    print("correlation_info running ... ")
    df_all_data = datamatrix
    corr_matrix = df_all_data.iloc[:,0:(df_all_data.shape[1]-1)].corr()
    sns.heatmap(corr_matrix,xticklabels=corr_matrix.columns,yticklabels=corr_matrix.columns)
    cormat_melted = []
    for i in range(len(corr_matrix)):
        f1 = corr_matrix.columns[i]
        for j in range(i,len(corr_matrix)):
            f2 = corr_matrix.columns[j]
            cormat_melted.append([f1, f2, corr_matrix.iloc[i,j]])
    cormat_melted = pd.DataFrame(cormat_melted,columns=['f1','f2','values'])
    cormat_melted.head(5)
    cormat_melted_filt = cormat_melted.loc[(cormat_melted['values']>=th) & (cormat_melted['values'] !=1.0)]
    todrop = set(cormat_melted_filt['f2'])
    #df_all_data.drop(todrop, axis=1, inplace=True)

    print ("Correlation filter >" , str(th) , ": " , str(len(todrop)) , " features from the dataset")
    print (todrop)

def naivebayes(df_data,cv):
    print("naivebayes running ... ")
    kf = KFold(n_splits=cv, random_state=0)
    result = []
    for train, test in kf.split(df_data):
        train_data = df_data.iloc[train,:]
        test_data =  df_data.iloc[test,:]

        trainx = train_data.iloc[:,0:df_data.shape[1]-2]
        trainy =   train_data.iloc[:,df_data.shape[1]-1]
        testx = test_data.iloc[:,0:df_data.shape[1]-2]
        testy = test_data.iloc[:,df_data.shape[1]-1]

        clf = GaussianNB()
        clf.fit(trainx, trainy.values)

        yhat = pd.DataFrame(clf.predict(testx), columns=['predict'])
        result.append(np.sum([1 if x == y else 0 for x, y in zip(testy.values, yhat.values)]) / float(len(yhat)))

    print ("Average naivebayes accuracy is:", np.sum(result)/len(result))
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pylab as plt

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

def logregression(df_data, cv_):
    predicted = cross_val_predict(LogisticRegression(), df_data.iloc[:,0:df_data.shape[1]-2], df_data.iloc[:,df_data.shape[1]-1], cv=cv_)
    print("Logistic regression score",metrics.accuracy_score(df_data.iloc[:,df_data.shape[1]-1], predicted))
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(df_data.iloc[:,df_data.shape[1]-1], predicted)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def grid_search(model , df_data, cv_, first_dim,verbose, second_dim=None, third_dim=None):
    if model == 'LRLASSO':
        auc_matrix = np.zeros(len(first_dim))
        for index, regularization_strength in enumerate(first_dim):
            model = LogisticRegression(penalty='l1', C=regularization_strength)
            predicted = cross_val_predict(model, df_data.iloc[:, 0:df_data.shape[1] - 2], df_data.iloc[:, df_data.shape[1] - 1], cv=cv_)
            auc_matrix[index] = metrics.accuracy_score(df_data.iloc[:,df_data.shape[1]-1], predicted)
            if verbose == True:
                print('GRID SEARCHING LR: progress: {0:.3f} % ...'.format((index + 1) / (len(first_dim)) * 100))
        return auc_matrix
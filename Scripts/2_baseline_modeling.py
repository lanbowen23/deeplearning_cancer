import pandas as pd
import importlib
importlib.reload(baseline_models)
from Scripts import baseline_models

df_all_data = pd.read_csv("~/ghub/deeplearning_cancer/Data/all_data_combined.csv")
df_all_data.head(10)

# check correlation
baseline_models.correlation_info(df_all_data.loc[:,'Leukemia_':'Liver_car'],0.7)
#Correlation filter > 0.7 :  1  features from the dataset

# naive bayes model 3 fold cv
baseline_models.naivebayes(df_all_data.loc[:,'Leukemia_':'class'],3)
#Average naivebayes accuracy is: 0.50395133322


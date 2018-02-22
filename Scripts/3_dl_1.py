from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pylab as plt



## experimental data
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



np.random.seed(7)



# create model
model = Sequential()
model.add(Dense(166, input_dim=166, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(df_all_data.iloc[:,0:df_all_data.shape[1]-1], df_all_data.iloc[:,df_all_data.shape[1]-1], epochs=10, batch_size=10)

# evaluate the model
scores = model.evaluate(df_all_data.iloc[:,0:df_all_data.shape[1]-1], df_all_data.iloc[:,df_all_data.shape[1]-1])
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# calculate predictions
predictions = model.predict(df_all_data.iloc[:,0:df_all_data.shape[1]-1])

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from random import randrange
import glob


allfiles = glob.glob("Data/*.csv")
print (allfiles)


dna_cordinates = pd.DataFrame()
counter =1
for file in allfiles:
    df = pd.read_csv(file,index_col=None, header=0)
    data_ids = df.iloc[:,[4]]
    data_ids.columns =['GENEID']

    dna_cordinates[file] = data_ids

    #print(data_ids.shape, dna_cordinates.shape)

print (dna_cordinates.head(3))
# get unique values form dna_cordinates datatframe
dna_cordinates_unique_ids = pd.Series(dna_cordinates.values.ravel()).unique()



all_data = pd.DataFrame()
all_data["GENEID"] = dna_cordinates_unique_ids
all_data.set_index("GENEID")

# construct a datamatrix
i = 1
for file in allfiles:
    print (file, i )
    i += 1
    name = file[5:14]
    all_data[name+"_Cancer"] =  0.0
    all_data[name + "_Normal"] = 0.0

    file_df = pd.read_csv(file,index_col=None, header=0)
    file_data = file_df.iloc[:,[4,5,6]]
    file_data.columns = ["GENEID", "EXP1", "EXP2"]

    visited_index =[]


    for i,row in file_data.iterrows():

        if i not in visited_index:

            replicates = file_data.loc[file_data["GENEID"] == row["GENEID"], :].index
            visited_index.append(x for x in replicates )

            exp1 = file_data.loc[file_data["GENEID"]==row["GENEID"],:].mean()[0]
            exp2 = file_data.loc[file_data["GENEID"] == row["GENEID"], :].mean()[1]


            all_data.loc[all_data["GENEID"]==row["GENEID"],name+"_Cancer"] = exp1
            all_data.loc[all_data["GENEID"] == row["GENEID"], name + "_Normal"] = exp2


        #print (all_data.loc[all_data["GENEID"]==row["GENEID"],:] )



all_data.to_csv("temp.csv")

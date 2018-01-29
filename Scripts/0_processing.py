import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from random import randrange
import glob


allfiles = glob.glob("Data/*.csv")
allfiles
frame = pd.DataFrame()
list_ = []
for file_ in allfiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    print (df.shape)
    print (df.head(2))
    print (file_)
    name = file_[5:14]
    df_2= df.loc[:, ["Transcript ID", "EXP1", "EXP2"]]
    df_2.columns = [name+"ID", name+"Cancer", name+"Normal"]
    print(df_2.head(2))
    break

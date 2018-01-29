import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from random import randrange


dat1 = pd.read_csv("Cancer_1_a_Breast_Carcinoma_pc.csv")
dat1.head(2)

dat1 = dat1.loc[:,["Transcript ID","EXP1","EXP2"]]
dat1.head(2)


dat12 = pd.read_csv("Cancer_1_b_Glioma_pc.csv")
dat12.head(2)

dat12 = dat12.loc[:,["Transcript ID","EXP1","EXP2"]]
dat12.columns = ["Transcript ID","L_EXP1","L_EXP2"]
dat12.head(2)

datc = pd.merge(dat1, dat12, on='Transcript ID',how='left')

datc.head(500)

rnaseq = dat1["Transcript ID"].values
lnc = dat12["Transcript ID"].values

i=0
for x in rnaseq:
    if x in lnc:
        i += 1
        print (i)
    break

common = [1 for x in rnaseq if x in lnc]

len(common)
len(rnaseq)
len(lnc)
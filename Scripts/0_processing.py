import pandas as pd
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
counter = 1
for file in allfiles:
    print (file, counter )
    counter += 1
    name = file[5:14]
    all_data[name+"_Cancer"] =  0.0
    all_data[name + "_Normal"] = 0.0

    file_df = pd.read_csv(file,index_col=None, header=0)
    file_data = file_df.iloc[:,[4,5,6]]
    file_data.columns = ["GENEID", "EXP1", "EXP2"]

    visited_index =[]


    for i,row in file_data.iterrows():

        if i not in visited_index:

            replicates = file_data.loc[file_data["GENEID"] == row["GENEID"], :].index.values
            visited_index.extend([x for x in replicates ])


            exp1 = file_data.loc[file_data["GENEID"]==row["GENEID"],:].mean()[0]
            exp2 = file_data.loc[file_data["GENEID"] == row["GENEID"], :].mean()[1]


            all_data.loc[all_data["GENEID"]==row["GENEID"],name+"_Cancer"] = exp1
            all_data.loc[all_data["GENEID"] == row["GENEID"], name + "_Normal"] = exp2


        #print (all_data.loc[all_data["GENEID"]==row["GENEID"],:] )



#all_data.to_csv("temp.csv")

all_data_normal = all_data.iloc[:,[x for x in range(62) if x%2 == 0]]
all_data_normal['class']=0
all_data_normal.to_csv("all_data_normal.csv")

caner_index =[0]
caner_index.extend([x for x in range(61) if x%2 != 0])
all_data_cancer = all_data.iloc[:,caner_index]
all_data_cancer['class']=1
all_data_cancer.to_csv("all_data_cancer.csv")


all_data_cancer.columns = ['GENEID', 'Leukemia_', 'Sarcoma_l', 'Lung_squa',
       'Glioblast', 'Ovarian_S', 'Cervical_',
       'Lung_aden', 'Testicula', 'Mesotheli',
       'Stomach_C', 'Sarcoma_p', 'Renal_Cel',
       'Colorecta', 'Esophagea', 'Pheochrom',
       'Colon_Car', 'Thyroid_c', 'Uveal_mel',
       'Pancreati', 'Cutaneous', 'Breast_Ca',
       'Endometri', 'Squamous_', 'Carcinoma',
       'Adrenocor', 'Malignant', 'Glioma_ln',
       'Glioma_pc', 'Prostate_', 'Liver_car', 'class']


all_data_normal.columns = ['GENEID', 'Leukemia_', 'Sarcoma_l', 'Lung_squa',
       'Glioblast', 'Ovarian_S', 'Cervical_',
       'Lung_aden', 'Testicula', 'Mesotheli',
       'Stomach_C', 'Sarcoma_p', 'Renal_Cel',
       'Colorecta', 'Esophagea', 'Pheochrom',
       'Colon_Car', 'Thyroid_c', 'Uveal_mel',
       'Pancreati', 'Cutaneous', 'Breast_Ca',
       'Endometri', 'Squamous_', 'Carcinoma',
       'Adrenocor', 'Malignant', 'Glioma_ln',
       'Glioma_pc', 'Prostate_', 'Liver_car', 'class']

all_data_combined = all_data_normal.append(all_data_cancer)
all_data_combined.shape
all_data_combined = all_data_combined.sample(frac=1).reset_index(drop=True)
all_data_combined.to_csv("all_data_combined.csv",index=False)
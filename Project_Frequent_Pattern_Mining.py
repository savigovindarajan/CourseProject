import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

dlbp_data =pd.read_csv("data/DBLP_Dataset.csv")
author_title = dlbp_data

author_title.drop(['conference','venue','pages','year','type','key','ee','url','MergedAuthors'], axis=1, inplace=True)
author = author_title.drop(author_title.columns[[0, 1]], axis=1)

dataset = author.to_numpy()


cleanedList=[]
for i in range(0,len(dataset)):
    temp = []
    for j in range(0,len(dataset[i])):
        if dataset[i][j] == dataset[i][j]:
            temp.append(dataset[i][j])
    cleanedList.append(temp)

#dataset= dataset[~np.isnan(dataset).any(axis=1)]
#dataset = dataset[np.logical_not(np.isnan(dataset))]
#print(cleanedList)

te = TransactionEncoder()
te_ary = te.fit(cleanedList).transform(cleanedList)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent = fpgrowth(df, min_support=0.001, use_colnames=True)
frequent = frequent[frequent['itemsets'].str.len()>1]


supp = frequent.support.unique()  # all unique support count
# Dictionay storing itemset with same support count key
freq_dic = {}
for i in range(len(supp)):
    inset = list(frequent.loc[frequent.support == supp[i]]['itemsets'])
    freq_dic[supp[i]] = inset
# Dictionay storing itemset with  support count <= key
freq_dic2 = {}
for i in range(len(supp)):
    inset2 = list(frequent.loc[frequent.support <= supp[i]]['itemsets'])
    freq_dic2[supp[i]] = inset2
# Find Closed frequent itemset

close_freq = []
for index, row in frequent.iterrows():
    isclose = True
    cli = row['itemsets']
    cls = row['support']
    checkset = freq_dic[cls]
    for i in checkset:
        if (cli != i):
            if (frozenset.issubset(cli, i)):
                isclose = False
                break

    if (isclose):
        close_freq.append(row['itemsets'])


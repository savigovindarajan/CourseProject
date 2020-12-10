import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import fnmatch

dlbp_data =pd.read_csv("data/DBLP_Dataset.csv")
author_title = dlbp_data

author_title.drop(['conference','venue','pages','year','type','key','ee','url','MergedAuthors'], axis=1, inplace=True)
#print(author_title)
author = author_title.drop(author_title.columns[[0, 1]], axis=1)

dataset = author.to_numpy()
#print(dataset)

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
#print(len(frequent['itemsets']))
frequentlist = list(frequent['itemsets'])


#print(cleanedList)
#print(mergedauthorlist)
#new_tran=[]
new_author_list = []
for i in range(0,len(frequentlist)):
    temp_author_list = []
    authorlist = list(frequentlist[i])
    found = 0
    for k in range(0,len(cleanedList)):
        #print(len(mergedauthorlist))
        #print(val,k)
        for j in range(0, len(authorlist)):
          #  print(authorlist[j])
         #   print(mergedauthorlist[k])
          #  print(authorlist[j])
            if (authorlist[j] in(cleanedList[k])):
                found = 1
            else:
                found = 0
                break

                #break
        if found == 1:
            for jj in range(0,len(authorlist)):
                if (authorlist[jj] in(cleanedList[k])):
                    cleanedList[k].remove(authorlist[jj])
            temp_author_list.append(cleanedList[k])

    new_author_list.append(temp_author_list)

#print(new_author_list)
context_indicator_list = []
for i in range(0,len(new_author_list)):
    te = TransactionEncoder()
    te_ary = te.fit(new_author_list[i]).transform(new_author_list[i])
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_author_list = fpgrowth(df, min_support=0.5, use_colnames=True)
   # print(i,frequent_author_list)

    supp = frequent_author_list.support.unique()  # all unique support count
    # Dictionay storing itemset with same support count key
    freq_dic = {}
    for i in range(len(supp)):
        inset = list(frequent_author_list.loc[frequent_author_list.support == supp[i]]['itemsets'])
        freq_dic[supp[i]] = inset
    # Dictionay storing itemset with  support count <= key
    freq_dic2 = {}
    for i in range(len(supp)):
        inset2 = list(frequent_author_list.loc[frequent_author_list.support <= supp[i]]['itemsets'])
        freq_dic2[supp[i]] = inset2
    # Find Closed frequent itemset

    close_freq = []
    for index, row in frequent_author_list.iterrows():
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
            close_freq.append([x for x in  (row['itemsets'])])
    context_indicator_list.append(close_freq)


for i in range(0,len(context_indicator_list)):
    print(context_indicator_list[i])

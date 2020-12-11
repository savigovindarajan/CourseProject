#pip install -i https://test.pypi.org/simple/ krovetz
#pip install -U prefixspan
from csv import reader
import pandas as pd
import krovetz as ks
import re
from prefixspan import PrefixSpan
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

stopwordlist = list()
stopwordlist = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about',
                'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 
                'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself',
                'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 
                'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don',
                'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 
                'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at',
                'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 
                'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not',
                'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which',
                'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 
                'doing', 'it', 'how', 'further', 'was', 'here', 'than','-','.']

def main():

    dlbp_data = pd.read_csv (r'DBLP_Dataset.csv',encoding="ISO-8859-1")  
    author_title = dlbp_data
    dataset = author_title.to_numpy()
    list1 = dataset[:,2].tolist()
    #convert authors to lower case
    list2 = []
    for i in list1:
        sublist = i.lower().split()
        list2.append(sublist)
    
    te = TransactionEncoder()
    te_ary = te.fit(list2).transform(list2)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent = fpgrowth(df, min_support=0.001, use_colnames=True)
    frequent = frequent[frequent['itemsets'].str.len()>1]

    frequent['itemsets']
    freqauth_list = []
    for i in frequent['itemsets']:
        freqauth_list.append([x for x in i])
    
    freqauth_dict = {}

    for i in freqauth_list:
        title_idx_sublist = []
        for idx, j in enumerate(list2):
            if set(i).issubset(j):
                title_idx_sublist.append(idx)
        freqauth_dict.update({tuple(i):title_idx_sublist})

    freqauth_title_dict = {}
    kstem = ks.PyKrovetzStemmer()
    for key, value in freqauth_dict.items():
        title_df = author_title.iloc[value]['title']
        title_sublist = list(title_df)
        title_sublists = []
        temp_list = []
        for x in title_sublist:
            tempx     = re.sub(r'[.]','', x)
            temp_list = re.sub(r'[^\x00-\x7F]+','', tempx).lower().split()
            temp_list2 = []
            if isinstance(temp_list, list):
                temp_list2.append([kstem.stem(z) for z in temp_list if not z in stopwordlist])
                title_sublists.extend(temp_list2)
            else:
                if not temp_list in stopwordlist:
                    title_sublists.extend([kstem.stem(temp_list)])
        freqauth_title_dict.update({key:title_sublists})

    # Closed / topk titles of frequent authors
    freqauth_title_dict_closed = {}

    for k, v in freqauth_title_dict.items():
        ps = PrefixSpan(v)
        closed_Seq_pattern = ps.topk(5, closed=True)
        freqauth_title_dict_closed.update({k:closed_Seq_pattern})

    # To get frequent author's context indicators
    frequentlist = freqauth_list
    cleanedList  = list2

    new_author_list = []
    for i in range(0,len(frequentlist)):
        temp_author_list = []
        authorlist = list(frequentlist[i])
        found = 0
        for k in range(0,len(cleanedList)):
            #print(len(mergedauthorlist))
            #print(val,k)
            for j in range(0, len(authorlist)):
                if (authorlist[j] in(cleanedList[k])):
                    found = 1
                else:
                    found = 0
                    break
                    
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
    
    freqauth_context_ind_dict = {}
    for authpair, titlelist in freqauth_title_dict_closed.items():
        cleantitlelist = []
        for i in titlelist:
            if isinstance(i, tuple):
                if isinstance(i[1], list):
                    listtostring = ' '.join(i[1])
                    cleantitlelist.append(listtostring)
        freqauth_context_ind_dict.update({authpair:cleantitlelist})

    # Merging both titles and Context indicator author for frequent pattern authors 
    for idx, key in enumerate(freqauth_context_ind_dict):
        newval = []
        if len(context_indicator_list[idx])> 0:
            for i in context_indicator_list[idx]:
                if len(i) > 0:                
                    tempstr = '&'.join(i)
                    #print(tempstr)
                    newval = freqauth_context_ind_dict[key]
                    newval.append(tempstr)
                    #print(newval)
                    freqauth_context_ind_dict.update({key:newval})

if __name__ == "__main__":
    main()
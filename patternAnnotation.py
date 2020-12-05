#pip install -i https://test.pypi.org/simple/ krovetz
#pip install -U prefixspan
from csv import reader
import pandas as pd
import krovetz as ks
import re
from prefixspan import PrefixSpan

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

    # with open('D:\\UIUC Course Work\\CS 410 Text Info Systems\\Project\\Merged_Clean_Dataset.csv','r',encoding='utf-8',errors='ignore') as read_csv:
    #     csv_reader = reader(read_csv)
    #     list_of_rows = list(csv_reader)
    #     read_csv.close()
    data = pd.read_csv (r'D:\\UIUC Course Work\\CS 410 Text Info Systems\\Project\\Merged_Clean_Dataset.csv',encoding="ISO-8859-1")  
    df = pd.DataFrame(data, columns= ['title'])
    print(df.head())
    kstem = ks.PyKrovetzStemmer()
    df['title'] = df['title'].str.split()
    pd.set_option('display.max_colwidth', -1)
    df['title'] = df['title'].apply(lambda x: [re.sub(r'[^\x00-\x7F]+','', y) for y in x])
    df['cleanTitle'] = df['title'].apply(lambda x: [kstem.stem(y) for y in x])
    df['cleanTitle'] = df['cleanTitle'].apply(lambda x: [word for word in x if not word in stopwordlist])
    print(df.head(15))

    ps = PrefixSpan(df['cleanTitle'])

    print(ps.frequent(10, closed=True))

if __name__ == "__main__":
    main()
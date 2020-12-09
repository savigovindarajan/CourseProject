from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from sklearn.feature_selection import mutual_info_classif

'''
This script can be used to calculate the similarity and weightage between two item sets (data type list) in 
terms of cosine similarity and mutual information/information gain. TF-IDF Weighting normalization has been 
used before converting them into vectors for more accurate result. It's expected to received the stemmed terms.
'''

def mutual_info(list1, list2):
    context1 = list1
    context2 = list2

    listToStrContext1 = ' '.join(map(str, context1))
    listToStrContext2 = ' '.join(map(str, context2))

    corpus = [listToStrContext1, listToStrContext2]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    #print(tfidf_vectorizer.get_feature_names())
    # print(tfidf_matrix)

    ndarray = tfidf_matrix.toarray()
    listOflist = ndarray.tolist()
    array1 = listOflist[0]
    array2 = listOflist[1]

    print(array1)
    print(array2)

    mutualInfo = mutual_info_score(array1, array2)
    return mutualInfo


def cos_similarity(list1, list2):
    context1 = list1
    context2 = list2

    listToStrContext1 = ' '.join(map(str, context1))
    listToStrContext2 = ' '.join(map(str, context2))

    corpus = [listToStrContext1, listToStrContext2]

    #print(corpus)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    # print(tfidf_matrix.shape)
    #print(tfidf_vectorizer.get_feature_names())
    # print(tfidf_matrix)

    cos_sim_array = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cos_sim_array[0, 1]

a = ["Deep", "Learning", "Algorithm", "Implementation", "Data", "Mining"]
b = ["Data", "Science", "Machine", "Learning", "Algorithm", "Implementation"]
# b = ["Dear", "Human", "Thakur", "House", "Midland", "Network"]

print("Cosine Similarity: ", cos_similarity(a, b))
print("Mutual Information: ", mutual_info(a, b))

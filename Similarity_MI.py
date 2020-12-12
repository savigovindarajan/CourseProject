from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
# from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer

'''
This script can be used to calculate the similarity and weightage between two item sets (data type list) in 
terms of cosine similarity and mutual information/information gain. TF-IDF Weighting normalization has been 
used before converting them into vectors for more accurate result. It's expected to received the stemmed terms.
'''

def mutual_info_tfidf(list1, list2):
    context1 = list1
    context2 = list2

    listToStrContext1 = ' '.join(map(str, context1))
    listToStrContext2 = ' '.join(map(str, context2))

    corpus = [listToStrContext1, listToStrContext2]

    # Build Vectors according to TF-IDF weighting
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    #print(tfidf_vectorizer.get_feature_names())
    #print(tfidf_matrix)

    ndarray = tfidf_matrix.toarray()
    listOflist = ndarray.tolist()
    array1 = listOflist[0]
    array2 = listOflist[1]

    #print(array1)
    #print(array2)

    mutualInfo = mutual_info_score(array1, array2)
    return mutualInfo


def mutual_info_count(list1, list2):
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    context1 = list1
    context2 = list2

    listToStrContext1 = ' '.join(map(str, context1))
    listToStrContext2 = ' '.join(map(str, context2))

    corpus = [listToStrContext1, listToStrContext2]

    # Build vectors according to the Count of the terms
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    #print(count_matrix)

    ndarray = count_matrix.toarray()
    listOflist = ndarray.tolist()
    array1 = listOflist[0]
    array2 = listOflist[1]

    print(array1)
    print(array2)

    mutualInfo = mutual_info_score(array1, array2)
    return mutualInfo

    '''
    normal_mutualInfo = normalized_mutual_info_score(array1, array2)
    print("Normalized Mutual info:", normal_mutualInfo)
    

    # Handle the scenario when the random variables are independent
    cos_sim_array = cosine_similarity(count_matrix, count_matrix)
    #if cos_sim_array[0, 1] == 0.0:
        #return 0.01
    #else:
    return mutualInfo
    '''




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
    # print(tfidf_vectorizer.get_feature_names())
    # print(tfidf_matrix)

    cos_sim_array = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cos_sim_array[0, 1]


a = ["Guillaume_Cabanac","enhanced information retrieval", "workshop"]
b = ["Data", "Science", "Machine", "Learning", "Algorithm", "Implementation"]






print("Cosine Similarity: ", cos_similarity(a, b))
print("Mutual Information as per TF-IDF: ", mutual_info_tfidf(a, b))
print("Mutual Information as per count: ", mutual_info_count(a, b))

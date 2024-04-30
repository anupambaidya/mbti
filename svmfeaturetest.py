import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from itertools import combinations 

def accu(test_y,y_pred):
    real_v = test_y.to_list()
    prv = list(y_pred)
    ac = 0
    for i in range(len(real_v)):
        if real_v[i] == prv[i]:
            ac += 1

    return(ac/len(real_v))

data = pd.read_csv("mbtipertweet_normalize.csv")
cls = pd.read_csv("classvalue.csv")

data = data.drop(columns=['Unnamed: 0'])
cls = cls.drop(columns=['Unnamed: 0'])

fea = [ 'url_no', 'emoji_n0','char_count', 'word_count', 'word_density',
       'punctuation_count', 'upper_case_word_count', 'stopwords_count',
       'unique_words_count', 'repeating_word_count', 'avg_word_length',
       'Adjctive_count', 'Verb_count', 'Adverb_count', 'Noun_count',
       'subjectivity_text', 'text_neg_score', 'text_neu_score',
       'text_pos_score', 'text_sentiment_score', 'neg_words_in_text',
       'pos_words_in_text', 'neu_words_in_text', 'text_polarity_Negetive',
       'text_polarity_Neutral', 'text_polarity_Positive']


tst = len(data) - int(len(data)/5)
print(tst)

fil = open("PCA.txt",'w')

for fe in range(len(fea)):
    #print(fe,(fea[0]))
    data1 = data.drop(columns=fea[fe])
    #print(fea[fe])

    train_x = data1[:tst]

    test_x = data1[tst :]

    train_y = cls['class'][:tst]
    test_y = cls['class'][tst:]

    clf = SVC()


    clf.fit(train_x, train_y)


    y_pred = clf.predict(test_x)

    #print(y_pred)
    result = accu(test_y, y_pred)
    fil.write("\nFeature excluded:  "+ fea[fe]+ "\t\t accuaccy :- " + str(result*100))
    print(fea[fe],"     :- ",result)

fel = list(range(len(fea)))

for fe in range(2, len(fea) - 5):
    #print(fe,(fea[0]))
    comb = combinations(fel, fe)
    print("number of feature : ",fe)
    fil.write("\n-------------------\nNumber of feature excluded : " + str(fe) + "\n")
    for i in list(comb): 
        exli = list(i)
        fil.write("\nFeature excluded:  ")
        data1 = data.copy()
        for cf in exli:
            data1 = data1.drop(columns=fea[cf])
            fil.write(fea[cf] + ', ')
        
   

        train_x = data1[:tst]

        test_x = data1[tst :]

        train_y = cls['class'][:tst]
        test_y = cls['class'][tst:]

        clf = SVC()


        clf.fit(train_x, train_y)


        y_pred = clf.predict(test_x)

        #print(y_pred)
        result = accu(test_y, y_pred)
        fil.write( "\t\t accuaccy :- " + str(result*100))
        print("     :- ",result)
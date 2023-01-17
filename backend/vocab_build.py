import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import nltk
from nltk.corpus import stopwords
import string
from MNB import MultinomialNaiveBayes
import pickle


def create_vocab(X_train) : 
    stop_words = list(stopwords.words('english'))

    vocab = dict() 
    cutoff_freq = 35
    features =[]
    

    for i in range(len(X_train)):
        for word in X_train[i].split():
            word_new  = word.strip(string.punctuation).lower()
            if (len(word_new)>2)  and (word_new not in stop_words):  
                if word_new in vocab:
                    vocab[word_new]+=1
                else:
                    vocab[word_new]=1
                
    for key in vocab:
        if vocab[key] >=cutoff_freq:
            features.append(key)
   
    return features


if __name__ == "__main__":
    df = pd.read_csv("dbpedia_8K.csv")

    categories = {
        'Company':0,
        'Education Institution':1,
        'Artist':2,
        'Athlete':3,
        'Office Holder':4,
        'Mean of Transportation':5,
        'Building':6,
        'Natural Place':7,
    }


    # 10-fold cross-validation
    kf = KFold(n_splits=10)

    ## creating training and testing data
    X = np.array(df['content'])
    Y = np.array(df['label'])
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=0)

    vocab = create_vocab(X_train)

    accuracy = np.array([])

    mnb = MultinomialNaiveBayes(vocab,categories,X_test,Y_test)

    i = 0
    for train_index, test_index in kf.split(X_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]
        print("running for fold:",i)
        i +=1
        mnb.train_model(x_train, y_train)
        test_class_pred = mnb.predict(x_test)
        accuracy_score = mnb.score(test_class_pred, y_test)
        accuracy = np.append(accuracy,accuracy_score)


    print("score on training data :",(np.average(accuracy))*100 )

    mnb.train_model(X_train, Y_train)
    Y_test_pred = mnb.predict(X_test)
    sklearn_score_test = mnb.score(Y_test_pred,Y_test)
    mnb.set_model_accuracy(sklearn_score_test)
    print("score on testing data :", mnb.get_model_accuracy())

    
    pickle.dump(mnb,open('mnb_classifier.pkl','wb'))
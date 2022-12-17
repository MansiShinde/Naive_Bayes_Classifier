from tkinter.tix import OptionMenu
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import string



# Implementing Multinomial Naive Bayes from scratch
class MultinomialNaiveBayes:
    
    def __init__(self,vocab,categories,X_test, Y_test):
        # count is a dictionary which stores several dictionaries corresponding to each news category
        # each value in the subdictionary represents the freq of the key corresponding to that news category 
        self.prior_prob = []
        self.total_words_in_each_class = []
        self.total_smooth_val_for_each_class = []
        self.vocab = vocab
        self.output = np.zeros((8, len(vocab)), dtype=int) 
        self.smoothing_matrix = np.full((8, len(vocab)), 1)
        self.model_accuracy = 0.0
        self.categories = categories
        self.top_10_words_prob = {}
        self.X_test = X_test
        self.Y_test = Y_test


    def calculate_priors(self,Y_train):
        for i in range(len(np.unique(Y_train))):
            no_of_doc = 0
            for j in Y_train:
                if j == i:
                    no_of_doc +=1
            self.prior_prob.append(no_of_doc / len(Y_train))

    

    def create_word_class_matrix(self,Y_train, X_train) : 

        index = []

        for i in range(len(np.unique(Y_train))):
            result = np.where(Y_train == i)
            index.append(result)

        for i in range(8):
            for j in index[i][0]:
                for word in X_train[j].split():
                    word_new  = word.strip(string.punctuation).lower()
                    if word_new in self.vocab:
                        ind = self.vocab.index(word_new)
                        self.output[i][ind] += 1


    def update_word_class_matrix(self,document,class_val) : 

        for word in document.split():
            word_new  = word.strip(string.punctuation).lower()
            if word_new in self.vocab:
                ind = self.vocab.index(word_new)
                self.output[class_val][ind] += 1
        


    def tot_words_in_each_class(self):
        self.total_words_in_each_class.clear()
        for i in range(8):
            tot_words = 0
            for j in range(len(self.vocab)):
                if self.output[i][j] > 0:
                    tot_words += self.output[i][j]
            self.total_words_in_each_class.append(tot_words)

    
    def update_smoothing_matrix(self):
    
        self.total_smooth_val_for_each_class.clear() 
        for i in range(8):
            tot_smoothing = 0
            for j in range(len(self.vocab)):
                tot_smoothing += self.smoothing_matrix[i][j]

            self.total_smooth_val_for_each_class.append(tot_smoothing) 

    
    def train_model(self,X_train, Y_train):
        self.calculate_priors(Y_train)
        self.create_word_class_matrix(Y_train, X_train)
        self.tot_words_in_each_class()
        self.update_smoothing_matrix()


    def predict(self,docs) :
        class_vals = []
        
        for doc in docs:
            store_probability_of_each_class = []
            for i in range(8):
                prob = 1.0
                for word in doc.split():
                    word_new  = word.strip(string.punctuation).lower()
                    if word_new in self.vocab:
                        val = ( (self.output[i][self.vocab.index(word_new)] + self.smoothing_matrix[i][self.vocab.index(word_new)]) / (self.total_words_in_each_class[i] + self.total_smooth_val_for_each_class[i]) )
                        prob = prob * val
                prob = prob * self.prior_prob[i]
                store_probability_of_each_class.append(prob)
                
            class_val = np.argmax(store_probability_of_each_class)
            class_vals.append(class_val)
            
        return class_vals


    def score(self,Y_pred,Y_true):
        # returns the mean accuracy
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_true[i]:
                count+=1
        return count/len(Y_pred)
    

    ### Calculate P(word/class) = (no of times word appears in class + laplace smoothing(1)) / (total no of words in class + total no of words in all the documents)
## further P(word/class) * P(class)

    def user_input_predict(self,doc):
               
        store_probability_of_each_class = []
        for i in range(8):
            prob = 1.0
            for word in doc.split():
                word_new  = word.strip(string.punctuation).lower()
                if word_new in self.vocab:
                    val = ( (self.output[i][self.vocab.index(word_new)] + self.smoothing_matrix[i][self.vocab.index(word_new)]) / (self.total_words_in_each_class[i] + self.total_smooth_val_for_each_class[i]) )
                    prob = prob * val
                else:
                    val = int(1) / (self.total_words_in_each_class[i] + self.total_smooth_val_for_each_class[i])
                    prob = prob * val
            prob = prob * self.prior_prob[i]
            store_probability_of_each_class.append(prob)   

        class_val = np.argmax(store_probability_of_each_class)

        class_accuracy = (store_probability_of_each_class[class_val] / sum(store_probability_of_each_class)) * 100

        return {list(self.categories.keys())[class_val] : class_accuracy}

    
    def get_top_10_words_contri_class(self,doc, category):
        top_10_words_prob = {}
        
        prob = 1.0 
        for word in doc.split():
            word_new  = word.strip(string.punctuation).lower()
            if word_new in self.vocab:
                val = ( (self.output[int(self.categories[category])][self.vocab.index(word_new)] + self.smoothing_matrix[int(self.categories[category])][self.vocab.index(word_new)]) / (self.total_words_in_each_class[int(self.categories[category])] + self.total_smooth_val_for_each_class[int(self.categories[category])] ))
                
                top_10_words_prob[word_new] = val
             
        return dict(sorted(top_10_words_prob.items(), key = itemgetter(1), reverse = True)[:10])
    

    def set_model_accuracy(self,score):
        self.model_accuracy = score


    def get_model_accuracy(self):
        return self.model_accuracy*100

    
    def do_model_prediction(self):
        Y_test_pred = self.predict(self.X_test)
        sklearn_score_test = self.score(Y_test_pred,self.Y_test)
        self.set_model_accuracy(sklearn_score_test)


    def remove_Word(self,word,category):
        removeWord = word
        class_val = self.categories[category]

        self.total_words_in_each_class[class_val] = self.total_words_in_each_class[class_val] - self.output[class_val][self.vocab.index(removeWord)]
        self.total_smooth_val_for_each_class[class_val] =  self.total_smooth_val_for_each_class[class_val] - self.smoothing_matrix[class_val][self.vocab.index(word)]
        self.smoothing_matrix[class_val][self.vocab.index(removeWord)] = 1
        self.total_smooth_val_for_each_class[class_val] =  self.total_smooth_val_for_each_class[class_val] + 1
        self.output[class_val][self.vocab.index(removeWord)] = 0
        

    def add_Word(self,word,category,doc):
        newWord = word
        class_val = self.categories[category]
        opt = dict()

        if newWord in self.vocab:

            self.update_word_class_matrix(doc,class_val)
            self.tot_words_in_each_class()

            opt = self.get_top_10_words_contri_class(doc,category)
            opt.popitem()
            val = ((self.output[class_val][self.vocab.index(newWord)] + self.smoothing_matrix[class_val][self.vocab.index(newWord)]) / (self.total_words_in_each_class[class_val] + self.total_smooth_val_for_each_class[class_val]))
            opt[newWord] = val

        else:

            add_col_output = np.zeros(8,dtype=int)
            add_col_smooth = np.full(8, 1)
            
            self.output = np.insert(self.output,len(self.output[0,:]),add_col_output,axis=1)
            self.smoothing_matrix = np.insert(self.smoothing_matrix, len(self.smoothing_matrix[0,:]),add_col_smooth, axis=1)
            
            self.vocab.append(newWord) 

            self.tot_words_in_each_class()
            self.update_smoothing_matrix()
            
            opt = self.get_top_10_words_contri_class(doc, category)
            opt.popitem()
            
            val = ( (self.output[class_val][len(self.output[0,:])-1] + self.smoothing_matrix[class_val][len(self.output[0,:])-1]) / (self.total_words_in_each_class[class_val] + self.total_smooth_val_for_each_class[class_val]))
            opt[newWord] = val

        return dict(sorted(opt.items(), key = itemgetter(1), reverse = True))
            

        
    def word_importance(self,prVal, upd_word, category):

        NewPrVal = prVal
        class_val = self.categories[category]
        word = upd_word

        self.total_smooth_val_for_each_class[class_val] =  self.total_smooth_val_for_each_class[class_val] - self.smoothing_matrix[class_val][self.vocab.index(word)]
            
        ## Calculating alpha value 
        alpha_for_word = 0.0
        alpha_for_word  = (NewPrVal*(self.total_smooth_val_for_each_class[class_val] + self.total_words_in_each_class[class_val]) - self.output[class_val][self.vocab.index(word)]) / (1 - NewPrVal)  

        ## update smoothing matrix and total_smoothing_val_for_each_class 
        self.smoothing_matrix[class_val][self.vocab.index(word)] = alpha_for_word
        self.total_smooth_val_for_each_class[class_val] =  self.total_smooth_val_for_each_class[class_val] + self.smoothing_matrix[class_val][self.vocab.index(word)]


            


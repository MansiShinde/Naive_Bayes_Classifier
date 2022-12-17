import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pickle
import json



app = Flask(__name__,template_folder='../templates')
model = pickle.load(open('mnb_classifier.pkl', 'rb'))
document = ''
category = ''
predictions = dict()
top_most_words = dict()


def denormalize_data(norm,data):
    d_min = min(data)
    d_max = max(data)
    d_mean = sum(data) / len(data)
    d_std = np.std(data)
    denorm = np.array(norm) * (d_max - d_min) + d_min
    return denorm



@app.route('/')
def home():
    model_accuracy = model.get_model_accuracy()
    predictions = {}
    keys = []
    values = []
    return render_template('index.html', model_score=model_accuracy,predictions = predictions ,labels=keys, values=values)



@app.route('/predict',methods=['POST'])
def predict():
    global predictions
    global document
    global top_most_words
    '''
    For rendering results on HTML GUI
    '''
    
    model_accuracy = model.get_model_accuracy()
    document = request.form["documentVal"]

    predictions = model.user_input_predict(document)
    category = list(predictions.keys())[0]
    top_most_words = model.get_top_10_words_contri_class(document, category)
    
    keys = list(top_most_words.keys())
    values = [ val for val in top_most_words.values()]
    normalized_values = preprocessing.normalize([values])
    return render_template('index.html', model_score=model_accuracy,predictions = predictions ,labels=keys, values=normalized_values.tolist()[0], document=document)



@app.route('/addword',methods=['POST'])
def addword():
    global predictions
    global top_most_words
    '''
    For rendering results on HTML GUI
    '''
    word = request.form["addword"]
    
    category = list(predictions.keys())[0]
    top_most_words = model.add_Word(word,category,document)
    keys = list(top_most_words.keys())
    values = [ val for val in top_most_words.values()]
    normalized_values = preprocessing.normalize([values])

    model.do_model_prediction()
    model_accuracy = model.get_model_accuracy()
    predictions = model.user_input_predict(document)

    return render_template('index.html', model_score=model_accuracy,predictions = predictions,labels=keys, values=normalized_values.tolist()[0],document=document)



@app.route('/removeword',methods=['POST'])
def removeword():
    global predictions
    global top_most_words
    '''
    For rendering results on HTML GUI
    '''
    word = request.form["removeword"]
    category = list(predictions.keys())[0]
    model.remove_Word(word,category)
    top_most_words = model.get_top_10_words_contri_class(document, category)
    keys = list(top_most_words.keys())
    values = [ val for val in top_most_words.values()]
    normalized_values = preprocessing.normalize([values])

    model.do_model_prediction()
    model_accuracy = model.get_model_accuracy()
    predictions = model.user_input_predict(document)

    return render_template('index.html',model_score=model_accuracy,predictions = predictions, labels=keys, values=normalized_values.tolist()[0],document=document)



@app.route('/wordImp/<string:value_to_send>',methods=['POST'])
def rwordImp(value_to_send):
    global predictions
    global top_most_words

    prVal = json.loads(value_to_send)
    word_index = prVal['index']    
    word_pr = prVal['value']
    category = list(predictions.keys())[0]
    word = list(top_most_words.keys())[word_index]

    values = [ val for val in top_most_words.values()]
    normalized_values = preprocessing.normalize([values])
    norm = normalized_values.tolist()[0]
    norm[word_index] = norm[word_index] + word_pr
    denorm = denormalize_data(norm,values)

    model.word_importance(denorm[word_index], word, category)
    model.do_model_prediction()
    model_accuracy = model.get_model_accuracy()
    predictions = model.user_input_predict(document)
    
    top_most_words = model.get_top_10_words_contri_class(document, category)
    keys = list(top_most_words.keys())
    values = [ val for val in top_most_words.values()]
    normalized_values = preprocessing.normalize([values])

    return render_template('index.html',model_score=model_accuracy,predictions = predictions, labels=keys, values=normalized_values.tolist()[0], document=document)



if __name__ == "__main__":

    app.run(debug=True)
# Naive_Bayes_Classifier

Implementing Naive Bayes Classifier from scratch to categorize documents.

Following are the files present in source code:

Backend: 
MNB.py
vocab_build.py 
app.py

Templates: 
index.html

1. MNB.py - this file has all the functions needed to implement MNB
2. vocab_build.py - this file creates a list of vocabulary needed to train MNB classifier. This
file will generate the pickle file needed to load in flask application
3. App.py - in this file, we load our model and connect it with frontend via Flask
4. Index.html - contains styling of the webpage along with displaying the results


The accuracy of the model on test data is: 93.06 %
While adding/removing or adjusting word importance, the model takes 15-20 secs to retrain the model and show the updated results.
Add/Remove Word Feature:
1. These features are available when the user right clicks on the bar chart.
2. On right clicking, he will see 2 options, “Add a Word”, “Remove a word”. Click on one
of them
3. If clicked on add word, pop up window opens asking for word addition and then hit
“ADD A WORD”
4. If the user just wants to exit the pop up without entering the word, he will do so by
clicking on the “ADD A WORD” button.
5. Follow the above steps for Remove Word
Word Importance Update:
1. To increase or decrease the word importance, the user can drag the bar of the
respective word whose importance he wishes to change, and then click on the update model button at the bottom of the window.





https://user-images.githubusercontent.com/29672533/213039359-7a4747d7-2126-499e-a8ef-c0a03e6f1c88.mov




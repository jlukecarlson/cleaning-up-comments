import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train_submission = train.copy()
test_submission = test.copy()


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Designing our baseline using word counts and a multi label naive bayes model
mdl = Pipeline([('vect', CountVectorizer()), 
                ('ovr_mnb', OneVsRestClassifier(MultinomialNB()))])

# Fitting the model to the training data
mdl.fit(train["comment_text"],train[labels])

# Predicting training data and reporting a score
train_submission[labels] = mdl.predict(train["comment_text"])
print("Training Total ROC AUC Avg:", roc_auc_score(train_submission[labels],train[labels]))

# Predicting test data - should be scored sparingly
test_submission[labels] = mdl.predict(test["comment_text"])

print("Test Total ROC AUC Avg:", roc_auc_score(test_submission[labels],test[labels]))


# Saving predictions
train_submission.to_csv("submissions/train_naive_bayes.csv")
test_submission.to_csv("submissions/test_naive_bayes.csv")
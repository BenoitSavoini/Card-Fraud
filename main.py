# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 18:05:23 2023

@author: Alkios
"""

import tensorflow as tf
print(tf.__version__)


tf.test.gpu_device_name()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, recall_score, f1_score



path = 'C:/Users/Alkios/Downloads/card_fraud/creditcard.csv'


def load_training_data(data_directory):
    df = pd.read_csv(data_directory, encoding = 'latin-1')
    return df



df = load_training_data(path)
df.columns


df['Class'].value_counts()
#highly imbalanced dataset


corr_table = df.corr()
corr_threshold = 0.1
target_col = 'Class'


corr_cols = corr_table.loc[(abs(corr_table[target_col]) > corr_threshold), target_col].index.tolist()

# Sélectionner les colonnes filtrées dans le dataframe
df_filtered = df[corr_cols]


X, y = df.iloc[:, :-1], df.iloc[:, -1:]


X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)


# Convert y_train and y_test to numpy arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()



#create two different dataframe of majority and minority class 
df_majority = df[(df['Class']==0)] 
df_minority = df[(df['Class']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 284315, # to match majority class
                                 random_state=42)  # reproducible results





from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
# Resampling the minority class. The strategy can be changed as required.
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_resample(df.drop('Class', axis=1), df['Class'])
oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)


X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(
     oversampled_X, oversampled_Y, test_size=0.33, random_state=42)

# Create a logistic regression object
logreg = LogisticRegression()

# Train the model using the training sets
logreg.fit(X_train_over, y_train_over)

# Predict the response for the test dataset
y_pred_over = logreg.predict(X_test_over)

accuracy_over = accuracy_score(y_test_over, y_pred_over)
recall_over = recall_score(y_test_over, y_pred_over)
f1_over = f1_score(y_test_over, y_pred_over)

print("Accuracy: {:.2f}".format(accuracy_over))
print("Recall: {:.2f}".format(recall_over))
print("F1 score: {:.2f}".format(f1_over))




from sklearn import svm
# Initialize SVM classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# Fit the classifier to the training data
clf.fit(X_train_over, y_train_over)

# Make predictions on the testing data
y_pred_over_svm = clf.predict(X_test_over)

accuracy_over_svm = accuracy_score(y_test_over, y_pred_over_svm)
recall_over_svm = recall_score(y_test_over, y_pred_over_svm)
f1_over_svm = f1_score(y_test_over, y_pred_over_svm)

print("Accuracy: {:.2f}".format(accuracy_over_svm))
print("Recall: {:.2f}".format(recall_over_svm))
print("F1 score: {:.2f}".format(f1_over_svm))



from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
#Create an instance
classifier = BalancedBaggingClassifier(estimator=DecisionTreeClassifier(),
                                sampling_strategy='not majority',
                                replacement=False,
                                random_state=42)
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)


y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: {:.2f}".format(accuracy))
print("Recall: {:.2f}".format(recall))
print("F1 score: {:.2f}".format(f1))
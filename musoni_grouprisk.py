# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:35:02 2016

@author: aggrey
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from tensorflow.contrib import skflow
import random
from sklearn.ensemble import RandomForestClassifier

# ----------- Helper Functions ----------
class dataDisplay(object):
    def __init__(self):
        pass
        
    def plot_principal(self, dataset):
        plt.hist(dataset['Average Principal'], alpha=0.8, bins=20)
        plt.title('Distribution of Average Principal')
        plt.xlabel('Average of days Overdue')
        plt.ylabel('Average Principal')
        plt.show()
    
    def plot_averageDaysoverdue(self, dataset):
        sns.kdeplot(dataset['Average Over Due Days'], shade=True, color='g')
        plt.title('Estimate if average overdue days as of December-15')
        plt.xlabel('Avrage of days overdue')
        plt.ylabel('Probability distribution')
        plt.show()
    
    def plot_averageAsOfLastMonth(self, dataset):
        sns.kdeplot(dataset['Past One Month'], shade=True, color = 'r')
        plt.title('Estimate of average overdue days as of past one month')
        plt.xlabel('Average of overdue days')
        plt.ylabel('Probability distribution')
        plt.show()

    def plot_averageAsOfPastThreeMonths(self, dataset):
        sns.kdeplot(dataset['Past Three Months'], shade=True, color='purple')
        plt.title('Estimate of average overdue days as of past three months')
        plt.xlabel('Averae of overdue days')
        plt.ylabel('Probability Distribution')
        plt.show()
    
    def plot_feature_importance(self, features, importance):
        
        indices = np.argsort(importances)
        plt.title('Feature Importances')
        plt.figure(1)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='darkorange', alpha = 1.0, align='center')
        plt.yticks(range(len(indices)), features[indices])
        plt.xlabel('Relative Importance')
        grp = plt.gcf()
        grp.savefig('visualizations/feature_importance.png', bbox_inches='tight')
        

random.seed(42)
#Load training data
df_train = pd.read_csv('musoni/groups_final_dataset.csv', low_memory=False)
train_data = df_train.drop(['Average Over Due Days'], axis = 1)
train_data = train_data.sample(frac = 1).reset_index(drop=True)
x_train = df_train[['Number Of Loans','Average Principal','Past One Month', 'Past Three Months']]
y_train = df_train[['Low/High Risk']]

viz = dataDisplay()
#viz.plot_principal(train_data)
#viz.plot_averageDaysoverdue(df_train)
#viz.plot_averageAsOfLastMonth(train_data)
#viz.plot_averageAsOfPastThreeMonths(train_data)

# Load testing data
df_test = pd.read_csv('musoni/test.csv', low_memory=False)
x_test = df_test[['Number Of Loans','Average Principal','Past One Month', 'Past Three Months']]
y_test = df_test[['Low/High Risk']]

classifier = RandomForestClassifier(n_estimators=1000)
classifier.fit(x_train, y_train.values.ravel())
output = classifier.predict(x_test)
score = metrics.accuracy_score(y_test, output)
_score = np.round(score, decimals = 1)
percentScore = _score * 100

print('\nAccuracy = ', percentScore)
df_test['Predicted Results'] = output
df_test.to_csv('musoni/predictions.csv', index=False)
print(df_test)

# Show which variables are more important than the other
features = df_train.columns[0:4]
importances = classifier.feature_importances_
viz.plot_feature_importance(features,importances)



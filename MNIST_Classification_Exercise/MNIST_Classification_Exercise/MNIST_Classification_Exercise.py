import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import time

def main():
    mnist = fetch_mldata('MNIST original')

    X, y = mnist['data'], mnist['target']


    #For testing purposes, remove later
    X = X[0:7000]
    y = y[0:7000]

    train_size = 6000
    test_size = (len(X) - train_size)/len(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=27)

    clfs = {"Logistic Regression": LogisticRegression(), "Decision Tree Classifier":tree.DecisionTreeClassifier()}

    y_train_0 = (y_train == 0)
    y_test_0 = (y_test == 0)

    for type in clfs:
        start_time = time.time()
        scores = cross_val_score(clfs[type], X_train, y_train_0, cv=5, scoring='accuracy')
        print("\nCross-validation scores for {}:\n".format(type))
        print(scores)
        y_pred = cross_val_predict(clfs[type], X_test, y_test_0)
        cm = confusion_matrix(y_test_0, y_pred)
        print("\nConfusion matrix for {}:\n".format(type))
        print(cm)
        rs = recall_score(y_test_0, y_pred)
        ps = precision_score(y_test_0, y_pred)
        print("Recall is {0:.2f} \tPrecision is: {0:.2f}\n".format(rs,ps))
        stop_time = time.time()
        delta = stop_time - start_time
        print("Elapsed time for determining data is {} seconds\n".format(delta))

if __name__ == '__main__':
    main()
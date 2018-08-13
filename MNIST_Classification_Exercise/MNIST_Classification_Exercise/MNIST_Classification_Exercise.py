import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve
import time

def main():
    '''
    Presents metrics and precision recall curves for finding zeros in the MNIST dataset.
    Compares Logistic Regression and SGD Classifier. Precision-Recall vs. Thresholds figures printed to pdf.
    '''
    mnist = fetch_mldata('MNIST original')

    X, y = mnist['data'], mnist['target']


    #For testing purposes, remove later
    X = X[0:7000]
    y = y[0:7000]

    train_size = 6000
    test_size = (len(X) - train_size)/len(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=27)

    clfs = {"Logistic Regression": LogisticRegression(),  "SGD Classifier":SGDClassifier(max_iter=1000, tol=1e-3, random_state = 0)}

    y_train_0 = (y_train == 0)
    y_test_0 = (y_test == 0)

    i = 1
    output_file = PdfPages('PrecisionRecallTradeoffCurves.pdf')

    for type in clfs:
        start_time = time.time()
        clfs[type].fit(X_train, y_train_0)
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

        y_scores = cross_val_predict(clfs[type], X_train, y_train_0, cv=5, method='decision_function')
        precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)
        plt.figure(num = i, figsize=(12,8)); 
        plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
        output_file.savefig()
        i = i+1

        stop_time = time.time()
        delta = stop_time - start_time
        print("Elapsed time for determining data is {} seconds\n".format(delta))

    output_file.close()

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([-0.5,1.5])    

if __name__ == '__main__':
    main()
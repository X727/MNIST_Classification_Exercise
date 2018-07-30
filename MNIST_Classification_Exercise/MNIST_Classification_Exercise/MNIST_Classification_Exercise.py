import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_score, recall_score

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
        scores = cross_val_score(clfs[type], X_train, y_train_0, cv=10, scoring='accuracy')
        print("Cross-validation scores for "+type+":\n")
        print(scores)
        y_pred = cross_val_predict(clfs[type], X_test, y_test_0)
        cm = confusion_matrix(y_test_0, y_pred)
        print("Confusion matrix for "+type+":\n")
        print(cm)
        rs = recall_score(y_test_0, y_pred)
        ps = precision_score(y_test_0, y_pred)
        print("Recall is :" + str(rs) + "\t Precision is: "+str(ps))




if __name__ == '__main__':
    main()
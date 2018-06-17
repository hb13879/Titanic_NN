import numpy as np
import pandas as pd
import NeuralNetwork

def read_in_data():
    return pd.read_csv("train.csv")

def drop_categories(train):
    return train.drop(["PassengerId","Survived","Name","Ticket","Cabin"],1)

def map_categories(train):
    sex_map = {"male" : 1, "female" : 0}
    embarked_map = {"S" : 0, "C" : 1, "Q" : 2}
    train["Sex"] = train["Sex"].map(sex_map)
    train["Embarked"] = train["Embarked"].map(embarked_map)

def complete_age_data(train):
    #find mean age
    mean_age = train["Age"].mean()
    train["Age"] = train["Age"].fillna(mean_age)
    train["Embarked"] = train["Embarked"].fillna(0)

def feature_normalisation(train):
    train["Age"] = (train["Age"]-train["Age"].mean())/train["Age"].std()
    train["Fare"] = (train["Fare"]-train["Fare"].mean())/train["Fare"].std()

def train_test_split(X,Y):
    X = X.values
    Y = Y.values
    Y = Y.reshape(np.shape(Y)[0],1)
    Y = np.transpose(Y)
    X = np.transpose(X)
    Xtrain = X[:,0:741]
    Xtest = X[:,741:]
    Ytrain = Y[:,0:741]
    Ytest = Y[:,741:]
    return Xtrain, Xtest, Ytrain, Ytest

def main():
    train = read_in_data()
    Ytrain = train["Survived"]
    Xtrain = drop_categories(train)
    map_categories(Xtrain)
    complete_age_data(Xtrain)
    feature_normalisation(Xtrain)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtrain,Ytrain)
    #nn = NeuralNetwork(Xtrain,Xtest,Ytrain,Ytest,layers=2,iterations = 1000,neurons=[1,1],activation_functions=[relu,sigmoid],dactivation_functions=[drelu,dsigmoid])
    nn = NeuralNetwork.NeuralNetwork(Xtrain,Xtest,Ytrain,Ytest,iterations=1000)
    nn.train_model()
    nn.test_model()

if __name__ == '__main__':
    main()

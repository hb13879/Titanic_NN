import numpy as np
import pandas as pd

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
    #find mean male an female ages
    #mean_ages = train.groupby("Sex")["Age"].mean()
    mean_age = train["Age"].mean()
    train["Age"] = train["Age"].fillna(mean_age)
    train["Embarked"] = train["Embarked"].fillna(0)

def feature_normalisation(train):
    train["Age"] = (train["Age"]-train["Age"].mean())/train["Age"].std()
    train["Fare"] = (train["Fare"]-train["Fare"].mean())/train["Fare"].std()

def relu(x):
    return x * (x>0)

def drelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def dsigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialise_weights(hidden_units,features,output_units):
    W1 = np.random.rand(hidden_units,features)
    W2 = np.random.rand(output_units,hidden_units)
    b1 = np.zeros((hidden_units,1))
    b2 = np.zeros((output_units,1))
    parameters = {"W1" : W1, "W2" : W2, "b1" : b1, "b2": b2}
    return parameters

def feedforward(Xtrain,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    z1 = np.dot(W1,Xtrain) + b1
    a1 = relu(z1)
    z2 = np.dot(W2,a1) + b2
    a2 = sigmoid(z2)
    cache = {"z1" : z1, "a1" : a1, "z2" : z2, "a2" : a2}
    return cache

def calculate_grads(cache,Ytrain,m,parameters,X):
    dA2 = (cache["a2"] - Ytrain)
    dW2 = (2/m) * np.dot((dsigmoid(cache["z2"]) * dA2), np.transpose(cache["a1"]))
    db2 = (2/m) * np.sum(dsigmoid(cache["z2"]) * dA2)
    dA1 = np.dot(parameters["W2"].T, dA2) * dsigmoid(cache["z2"])
    dW1 = (2/m) * np.dot((drelu(cache["z1"]) * dA1), np.transpose(X))
    db1 = (2/m) * np.sum(drelu(cache["z1"]) * dA1)
    grads = {"dW1" : dW1, "db1" : db1, "dW2" : dW2, "db2" : db2}
    return grads

def adjust_weights(parameters,alpha,grads):
    parameters["W2"] = parameters["W2"] - alpha * grads["dW2"]
    parameters["W1"] = parameters["W1"] - alpha * grads["dW1"]
    parameters["b1"] = parameters["b1"] - alpha * grads["db1"]
    parameters["b2"] = parameters["b2"] - alpha * grads["db2"]
    return parameters

def calculate_cost(a2,Ytrain,m):
    return np.sum(np.square(a2 - Ytrain))/m

def calculate_accuracy(a2,Ytrain,m):
    predictions = (a2 > 0.5)
    acc = (predictions == Ytrain)
    acc = np.sum(acc) / m
    return acc

def test_model(Xtest,Ytest,parameters,m):
    cache = feedforward(Xtest,parameters)
    print("Cost: " + str(calculate_cost(cache["a2"],Ytest,m)))
    print("Test Accuracy:" + str(calculate_accuracy(cache["a2"],Ytest,m)))

def train_neural_network(Xtrain,Ytrain,Xtest,Ytest):
    alpha = 0.5
    iterations = 10000
    mtrain = np.shape(Ytrain)[1]
    mtest = np.shape(Ytest)[1]
    parameters = initialise_weights(3,np.shape(Xtrain)[0],1)
    for i in range(0,iterations):
        cache = feedforward(Xtrain,parameters)
        grads = calculate_grads(cache,Ytrain,mtrain,parameters,Xtrain)
        parameters = adjust_weights(parameters,alpha,grads)
        print(calculate_cost(cache["a2"],Ytrain,mtrain))
    print("Training Accuracy:" + str(calculate_accuracy(cache["a2"],Ytrain,mtrain)))
    test_model(Xtest,Ytest,parameters,mtest)

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
    train_neural_network(Xtrain,Ytrain,Xtest,Ytest)

class NeuralNetwork(object):

    def __init__(self,alpha=0.1):
        pass




if __name__ == '__main__':
    main()

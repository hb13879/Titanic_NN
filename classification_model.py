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
    print(np.shape(dA2))
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
    #train_neural_network(Xtrain,Ytrain,Xtest,Ytest)
    nn = NeuralNetwork(Xtrain,Xtest,Ytrain,Ytest,iterations = 1)

class NeuralNetwork(object):

    parameters = []
    activation_functions = []
    dactivation = []
    cache = []
    dataset = {}
    neurons = [] #list containing number of neurons in each layer in order (excl. input layer)

    def __init__(self,Xtrain,Xtest,Ytrain,Ytest,layers=2,neurons=[3,1],alpha=0.1,iterations=1000,activation_functions=[relu,sigmoid],dactivation_functions=[drelu,dsigmoid]):
        self.alpha = alpha
        self.iterations = iterations
        self.layers = layers
        self.neurons = neurons
        self.activation_functions = activation_functions
        self.dactivation = dactivation_functions
        self.dataset["Xtrain"] = Xtrain
        self.dataset["Xtest"] = Xtest
        self.dataset["Ytrain"] = Ytrain
        self.dataset["Ytest"] = Ytest
        self.mtrain = np.shape(self.dataset["Xtrain"])[1]
        self.mtest = np.shape(self.dataset["Xtest"])[1]
        self.train_model()

    def __initialise_weights(self):
        a = np.shape(self.dataset["Xtrain"])[0]
        print(a)
        self.neurons = [a] + self.neurons
        for first,second in zip(self.neurons,self.neurons[1:]):
            w = np.random.rand(second,first)
            b = np.zeros((second,1))
            self.parameters.append(w)
            self.parameters.append(b)

    def __feedforward(self):
        z = []
        a = self.dataset["Xtrain"]
        a = [a] + []
        for l in range(self.layers):
            z.append(np.dot(self.parameters[2*l],a[l]) + self.parameters[2*l+1])
            a.append(self.activation_functions[l](z[l]))
        return a,z


    def __calculate_grads(self,a,z):

        dA = []
        dW = []
        db = []
        dA.append(a[2] - self.dataset["Ytrain"])
        #dA2 = (cache["a2"] - Ytrain)
        dW.append((2/self.mtrain) * np.dot((self.dactivation[1](z[1]) * dA[0]), a[1].T))
        #dW2 = (2/m) * np.dot((dsigmoid(cache["z2"]) * dA2), np.transpose(cache["a1"]))
        db.append((2/self.mtrain) * np.sum(self.dactivation[1](z[1]) * dA[0]))
        #db2 = (2/m) * np.sum(dsigmoid(cache["z2"]) * dA2)
        dA.append(np.dot(self.parameters[2].T, dA[0]) * dsigmoid(z[1]))
        #dA1 = np.dot(parameters["W2"].T, dA2) * dsigmoid(cache["z2"])
        dW.append((2/self.mtrain) * np.dot((self.dactivation[0](z[0]) * dA[1]), self.dataset["Xtrain"].T))
        #dW1 = (2/m) * np.dot((drelu(cache["z1"]) * dA1), np.transpose(X))
        db.append((2/self.mtrain) * np.sum(self.dactivation[0](z[0]) * dA[1]))
        #db1 = (2/m) * np.sum(drelu(cache["z1"]) * dA1)
        #grads = {"dW1" : dW1, "db1" : db1, "dW2" : dW2, "db2" : db2}
        return dA,dW,db


    def train_model(self):
        self.__initialise_weights()
        for i in range(self.iterations):
            a,z = self.__feedforward()
            dA,dW,db = self.__calculate_grads(a,z)
            #parameters = adjust_weights(parameters,alpha,grads)
            #print(calculate_cost(cache["a2"],Ytrain,mtrain))
        #print("Training Accuracy:" + str(calculate_accuracy(cache["a2"],Ytrain,mtrain)))
        #test_model(Xtest,Ytest,parameters,mtest)




if __name__ == '__main__':
    main()

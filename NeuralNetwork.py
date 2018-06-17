import numpy as np

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

class NeuralNetwork(object):
    
    dataset = {}

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
        self.parameters = []

    def __initialise_weights(self):
        a = np.shape(self.dataset["Xtrain"])[0]
        self.neurons = [a] + self.neurons
        for first,second in zip(self.neurons,self.neurons[1:]):
            w = np.random.rand(second,first)
            b = np.zeros((second,1))
            self.parameters.append(w)
            self.parameters.append(b)

    def __feedforward(self, train):
        z = []
        if train:
            a = self.dataset["Xtrain"]
        else:
            a = self.dataset["Xtest"]
        a = [a] + []
        for l in range(self.layers):
            z.append(np.dot(self.parameters[2*l],a[l]) + self.parameters[(2*l)+1])
            a.append(self.activation_functions[l](z[l]))
        return a,z


    def __calculate_grads(self,a,z,l):
        dA = []
        dW = []
        db = []
        for i in range(l):
            if i == 0:
                dA.append(a[l] - self.dataset["Ytrain"])
            dW.append((2/self.mtrain) * np.dot((self.dactivation[l-(i+1)](z[l-(i+1)]) * dA[i-1]), a[l-(i+1)].T))
            db.append((2/self.mtrain) * np.sum(self.dactivation[l-(i+1)](z[l-(i+1)]) * dA[i-1]))
            if i != 0:
                dA.append(np.dot(self.parameters[2*(l-i)].T, dA[i-1]) * self.dactivation[l-i](z[l-(i+1)]))
        return dW,db

    def __adjust_weights(self,dW,db):
        for i in range(self.layers):
            self.parameters[i*2] -= self.alpha*dW[-1-i]
            self.parameters[(i*2)+1] -= self.alpha*db[-1-i]

    def __calculate_cost(self,a,train):
        if train:
            y = self.dataset["Ytrain"]
            m = self.mtrain
        else:
            y = self.dataset["Ytest"]
            m = self.mtest
        return np.sum(np.square(a - y))/self.mtest

    def __calculate_accuracy(self,a,train):
        predictions = (a > 0.5)
        if train:
            y = self.dataset["Ytrain"]
            m = self.mtrain
        else:
            y = self.dataset["Ytest"]
            m = self.mtest
        acc = (predictions == y)
        acc = np.sum(acc) / m
        return acc

    def train_model(self):
        self.__initialise_weights()
        for i in range(self.iterations):
            a,z = self.__feedforward(train = True)
            dW,db = self.__calculate_grads(a,z,self.layers)
            self.__adjust_weights(dW,db)
            print(self.__calculate_cost(a[self.layers],train=True))
        print("Training Accuracy:" + str(self.__calculate_accuracy(a[self.layers],train=True)))

    def test_model(self):
        a,z = self.__feedforward(train=False)
        #print("Cost: " + str(self.__calculate_cost(a[self.layers],train=False)))
        print("Test Accuracy:" + str(self.__calculate_accuracy(a[self.layers],train=False)))

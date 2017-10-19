import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1 = 60 #num of neurons in hidden layer 1
no_hidden2 = 20 #num of neurons in next hidden layers
learning_rate = 10**-4

floatX = theano.config.floatX

# scale and normalize input data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max - X_min)

def normalize(X, X_mean, X_std):
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

#read and divide data into test and train sets
cal_housing = np.loadtxt('cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data
m = 3*X_data.shape[0] // 10
testX, testY = X_data[:m],Y_data[:m]
trainX, trainY = X_data[m:], Y_data[m:]

# scale and normalize data
trainX_max, trainX_min =  np.max(trainX, axis=0), np.min(trainX, axis=0)
testX_max, testX_min =  np.max(testX, axis=0), np.min(testX, axis=0)

trainX = scale(trainX, trainX_min, trainX_max)
testX = scale(testX, testX_min, testX_max)

trainX_mean, trainX_std = np.mean(trainX, axis=0), np.std(trainX, axis=0)
testX_mean, testX_std = np.mean(testX, axis=0), np.std(testX, axis=0)

trainX = normalize(trainX, trainX_mean, trainX_std)
testX = normalize(testX, testX_mean, testX_std)

no_features = trainX.shape[1]
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# initialize weights and biases for hidden layer(s) and output layer
w_o = theano.shared(np.random.randn(no_hidden2)*.01, floatX )
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)
w_h2 = theano.shared(np.random.randn(no_hidden1, no_hidden2)*.01, floatX )
b_h2 = theano.shared(np.random.randn(no_hidden2)*0.01, floatX)
w_h3 = theano.shared(np.random.randn(no_hidden2, no_hidden2)*.01, floatX )
b_h3 = theano.shared(np.random.randn(no_hidden2)*0.01, floatX)

init_w_o = w_o.get_value()
init_b_o = b_o.get_value()
init_w_h1 = w_h1.get_value()
init_b_h1 = b_h1.get_value()
init_w_h2 = w_h2.get_value()
init_b_h2 = b_h2.get_value()

# learning rate
alpha = theano.shared(learning_rate, floatX)

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
h2_out = T.nnet.sigmoid(T.dot(h1_out, w_h2) + b_h2)
h3_out = T.nnet.sigmoid(T.dot(h2_out, w_h3) + b_h3)
y4 = T.dot(h2_out, w_o) + b_o #output for 4 hidden layers
y5 = T.dot(h3_out, w_o) + b_o #output for 5 hidden layers

#we need to differentiate the expression for each network
cost4 = T.abs_(T.mean(T.sqr(d - y4)))
accuracy4 = T.mean(d - y4)
cost5 = T.abs_(T.mean(T.sqr(d - y5)))
accuracy5 = T.mean(d - y5)

#define gradients
dw_o, db_o, dw_h, db_h, dw_h2, db_h2 = T.grad(cost4, [w_o, b_o, w_h1, b_h1, w_h2, b_h2])
dw_o, db_o, dw_h, db_h, dw_h2, db_h2, dw_h3, db_h3 = T.grad(cost5, [w_o, b_o, w_h1, b_h1, w_h2, b_h2, w_h3, b_h3])

train4 = theano.function(
        inputs = [x, d],
        outputs = cost4,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h],
                   [w_h2, w_h2 - alpha*dw_h2],
                   [b_h2, b_h2 - alpha*db_h2]],
        allow_input_downcast=True
        )

train5 = theano.function(
        inputs = [x, d],
        outputs = cost5,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h],
                   [w_h2, w_h2 - alpha*dw_h2],
                   [b_h2, b_h2 - alpha*db_h2],
                   [w_h3, w_h3 - alpha*dw_h3],
                   [b_h3, b_h3 - alpha*db_h3]],
        allow_input_downcast=True
        )

test4 = theano.function(
    inputs = [x, d],
    outputs = [y4, cost4, accuracy4],
    allow_input_downcast=True
    )

test5 = theano.function(
    inputs = [x, d],
    outputs = [y5, cost5, accuracy5],
    allow_input_downcast=True
    )


alpha.set_value(learning_rate)
networks = [4,5]
n=len(trainX)

for i in range(len(networks)):
    train_cost = np.zeros(epochs)
    test_cost = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    w_o.set_value(init_w_o) #just needed for the second network
    b_o.set_value(init_b_o)
    w_h1.set_value(init_w_h1)
    b_h1.set_value(init_b_h1)
    w_h2.set_value(init_w_h2)
    b_h2.set_value(init_b_h2)

    min_error = 1e+15
    best_iter = 0
    for iter in range(epochs):
        if iter % 100 == 0:
            print(iter)
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            if i==0:
                cost += train4(trainX[start:end], np.transpose(trainY[start:end]))
            else:
                cost += train5(trainX[start:end], np.transpose(trainY[start:end]))
        train_cost[iter] = cost/(n // batch_size)
        if i==0:
            pred, test_cost[iter], test_accuracy[iter] = test4(testX, np.transpose(testY))
        else:
            pred, test_cost[iter], test_accuracy[iter] = test5(testX, np.transpose(testY))

        if test_cost[iter] < min_error:
            best_iter = iter
            min_error = test_cost[iter]
            best_w_o = w_o.get_value()
            best_b_o = b_o.get_value()
            best_w_h1 = w_h1.get_value()
            best_b_h1 = b_h1.get_value()
            best_w_h2 = w_h2.get_value()
            best_b_h2 = b_h2.get_value()
            if i==1:
                best_w_h3 = w_h3.get_value()
                best_b_h3 = b_h3.get_value()

    #set weights and biases to values at which performance was best
    w_o.set_value(best_w_o)
    b_o.set_value(best_b_o)
    w_h1.set_value(best_w_h1)
    b_h1.set_value(best_b_h1)
    w_h2.set_value(best_w_h2)
    b_h2.set_value(best_b_h2)
    if i==1:
        w_h3.set_value(best_w_h3)
        b_h3.set_value(best_b_h3)
        best_pred, best_cost, best_accuracy = test5(testX, np.transpose(testY))
    else:
        best_pred, best_cost, best_accuracy = test4(testX, np.transpose(testY))

    print(str(networks[i])+'-layers network : Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))

    plt.figure(1,figsize=(15,9))
    plt.subplot(121)
    plt.plot(range(epochs), test_accuracy)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Test accuracy for 4-layer and 5-layer networks')
    plt.subplot(122)
    plt.plot(range(epochs), test_accuracy)
    plt.axis([0, epochs, -10000, 20000])
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Test accuracy for 4-layer and 5-layer networks rescaled')
    plt.legend(['hidden layers = 4', 'hidden layers = 5'], loc='lower right')


plt.savefig('p_1b_4_5_layers_networks.png')

plt.show()

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import time

def init_bias(n = 1):
    return np.zeros(n)

def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(
        np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)),
        dtype=theano.config.floatX
        )
    if logistic == True:
        W_values *= 4
    return W_values


# scale data
def scale(X, X_min, X_max):
    return (X - X_min)/(X_max-np.min(X, axis=0))

# update parameters
def sgd(cost, params, lr=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


decay_array = [10**-3, 10**-6, 10**-9, 10**-12, 0]
learning_rate = 0.01
epochs = 10000
batch_size = 8
hidden_neurons = 15

# theano expressions
X = T.matrix() #features
Y = T.matrix() #output
decay = T.scalar()

w1_init = init_weights(36, hidden_neurons)
b1_init = init_bias(hidden_neurons) #weights and biases from input to hidden layer
w2_init = init_weights(hidden_neurons, 6, logistic=False)
b2_init = init_bias(6) #weights and biases from hidden to output layer

w1=theano.shared(value=w1_init, name='W', borrow=True)
b1=theano.shared(b1_init, theano.config.floatX)
w2=theano.shared(value=w2_init, name='W', borrow=True)
b2=theano.shared(b2_init, theano.config.floatX)

h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
py = T.nnet.softmax(T.dot(h1, w2) + b2)

y_x = T.argmax(py, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2))))
params = [w1, b1, w2, b2]
updates = sgd(cost, params, learning_rate)

# compile
train = theano.function(inputs=[X, Y, decay], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

#read train data
train_input = np.loadtxt('sat_train.txt',delimiter=' ')
trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)
trainX = scale(trainX, trainX_min, trainX_max)

train_Y[train_Y == 7] = 6
trainY = np.zeros((train_Y.shape[0], 6))
trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1


#read test data
test_input = np.loadtxt('sat_test.txt',delimiter=' ')
testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)

testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
testX = scale(testX, testX_min, testX_max)

test_Y[test_Y == 7] = 6
testY = np.zeros((test_Y.shape[0], 6))
testY[np.arange(test_Y.shape[0]), test_Y-1] = 1

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

# first, experiment with a small sample of data
##trainX = trainX[:1000]
##trainY = trainY[:1000]
##testX = testX[-250:]
##testY = testY[-250:]

# train and test
n = len(trainX)
time_for_updates = []

for j in range(len(decay_array)):
    test_accuracy = []
    train_cost = []
    dec = decay_array[j]
    w1.set_value(w1_init)
    b1.set_value(b1_init)
    w2.set_value(w2_init)
    b2.set_value(b2_init)
    for i in range(epochs):
        if i % 1000 == 0:
            print(i)
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end], dec)
        train_cost = np.append(train_cost, cost/(n // batch_size))

        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))
    print('decay = %.1g'%dec)
    print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))
    plt.figure(1,figsize=(15,9))

    plt.subplot(121)
    plt.plot(range(epochs), train_cost)
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')

    plt.subplot(122)
    plt.plot(range(epochs), test_accuracy)
    plt.axis([0, epochs, 0.3, 0.9])
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.legend(['decay = 10^-3', 'decay = 10^-6', 'decay = 10^-9', 'decay = 10^-12', 'decay = 0'], loc='lower right')
    plt.tight_layout()

#Plots
plt.savefig('p1a_accuracy_and_training_cost_decay_2.png')
plt.show()

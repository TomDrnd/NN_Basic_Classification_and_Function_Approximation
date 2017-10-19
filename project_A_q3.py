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


decay = 1e-6
learning_rate = 0.01
epochs = 10000
batch_size = 8
number_of_hiddenlayer_neuron = [5, 10, 15, 20, 25]

# theano expressions
X = T.matrix() #features
Y = T.matrix() #output

#starting vlaues for weights and biases from input to hidden layer
#here we initialize the maximum number of weights and biases that we can need and
#further we will select the right size that we need
w1_init = init_weights(36, max(number_of_hiddenlayer_neuron))
b1_init = init_bias(max(number_of_hiddenlayer_neuron))
#weights and biases from hidden to output layer
w2_init = init_weights(max(number_of_hiddenlayer_neuron), 6, logistic=False)
b2_init = init_bias(6)

#declaration of the theano variables for weights and biases
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
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
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


# train and test
n = len(trainX)
time_for_updates = []

for j in range(len(number_of_hiddenlayer_neuron)):
    test_accuracy = []
    train_cost = []
    w1.set_value(w1_init[0:36,0:number_of_hiddenlayer_neuron[j]])
    b1.set_value(b1_init[0:number_of_hiddenlayer_neuron[j]])
    w2.set_value(w2_init[0:number_of_hiddenlayer_neuron[j],0:6])
    b2.set_value(b2_init)
    t1 = time.time()
    for i in range(epochs):
        if i % 1000 == 0:
            print(i)
        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
        train_cost = np.append(train_cost, cost/(n // batch_size))

        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))
    t2 = time.time()
    print('number of hidden-layer neurons = %d'%number_of_hiddenlayer_neuron[j])

#Plots

    plt.figure(1,figsize=(15,9))

    plt.subplot(121)
    plt.plot(range(epochs), train_cost)
    plt.axis([0, epochs, 0, 0.8])
    plt.xlabel('iterations')
    plt.ylabel('cross-entropy')
    plt.title('training cost')

    plt.subplot(122)
    plt.plot(range(epochs), test_accuracy)
    plt.axis([0, epochs, 0.7, 0.95])
    plt.xlabel('iterations')
    plt.ylabel('accuracy')
    plt.title('test accuracy')
    plt.legend(['hidden-layer neurons = 5', 'hidden-layer neurons = 10', 'hidden-layer neurons = 15', 'hidden-layer neurons = 20', 'hidden-layer neurons = 25'], loc='lower right')
    plt.tight_layout()

    print('%.1f accuracy at %d iterations'%(np.max(test_accuracy)*100, np.argmax(test_accuracy)+1))
    number_of_updates_per_epoch = end/batch_size
    time_for_updates = np.append(time_for_updates, ((t2-t1)*1000000/epochs)/number_of_updates_per_epoch)


plt.savefig('p1a_accuracy_and_training_cost_number_of_neurons.png')

plt.figure()
plt.plot(number_of_hiddenlayer_neuron, time_for_updates)
plt.xlabel('number of hidden-layer neurons')
plt.ylabel('time for update in micro secondes')
plt.title('time for update at number of hidden-layer neurons')
plt.savefig('p1a_time_for_updates_per_number_of_hidden-layer_neurons.png')

plt.show()

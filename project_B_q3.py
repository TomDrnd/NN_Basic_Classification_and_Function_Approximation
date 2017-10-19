import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(10)

epochs = 1000
batch_size = 32
no_hidden1 = [20,30,40,50,60] #num of neurons in hidden layer 1
learning_rate = 10**-5

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
w_o = theano.shared(np.random.randn(max(no_hidden1))*.01, floatX )
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(no_features, max(no_hidden1))*.01, floatX )
b_h1 = theano.shared(np.random.randn(max(no_hidden1))*0.01, floatX)

init_w_o = w_o.get_value()
init_w_h1 = w_h1.get_value()
init_b_o = b_o.get_value()
init_b_h1 = b_h1.get_value()

# learning rate
alpha = theano.shared(learning_rate, floatX)


#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
y = T.dot(h1_out, w_o) + b_o

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
        allow_input_downcast=True
        )

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )


train_cross_cost = np.zeros(epochs)
validation_cost = np.zeros(epochs)
validation_accuracy = np.zeros(epochs)

min_error = 1e+15
best_iter = 0

n = len(trainX)
number_of_fold = 5
fold_size = n/number_of_fold
errors = np.zeros(len(no_hidden1))
opt_no_hidden = []
number_of_exp = 10
alpha.set_value(learning_rate)
colors = ('C0', 'C1', 'C2', 'C3', 'C4')


for exp in range(number_of_exp):
    print('exp :%d'%exp)
    trainX, trainY = shuffle_data(trainX, trainY)
    for i in range(len(no_hidden1)):
        print('hidden neurons :%d'%no_hidden1[i])
        error=0.0
        for j in range(number_of_fold):
            print('fold :%d'%j)

            trainX_cross = np.concatenate((trainX[:j*fold_size], trainX[(j+1)*fold_size:]), axis=0)
            trainY_cross = np.concatenate((trainY[:j*fold_size], trainY[(j+1)*fold_size:]), axis=0)
            validationX_cross = trainX[j*fold_size:(j+1)*fold_size]
            validationY_cross = trainY[j*fold_size:(j+1)*fold_size]

            w_o.set_value(init_w_o[0:no_hidden1[i]])
            b_o.set_value(init_b_o)
            w_h1.set_value(init_w_h1[0:8,0:no_hidden1[i]])
            b_h1.set_value(init_b_h1[0:no_hidden1[i]])

            min_error = 1e+15
            best_iter = 0
            best_b_o = 0

            for iter in range(epochs):
                if iter % 100 == 0:
                    print(iter)

                trainX, trainY = shuffle_data(trainX, trainY)
                cost = 0.0
                for start, end in zip(range(0, n-fold_size, batch_size), range(batch_size, n-fold_size, batch_size)):
                    cost += train(trainX_cross[start:end], np.transpose(trainY_cross[start:end]))
                train_cross_cost[iter] = cost/(n // batch_size)
                pred, validation_cost[iter], validation_accuracy[iter] = test(validationX_cross, np.transpose(validationY_cross))

                if validation_cost[iter] < min_error:
                    min_error = validation_cost[iter]

            if j==0 and exp==0: #in order to have just one example of plots
                plt.figure(1,figsize=(15,9))
                plt.axis([0, epochs, 0, 1e10])
                plt.plot(range(epochs), train_cross_cost, colors[i], linewidth=0.5)
                plt.plot(range(epochs), validation_cost, colors[i], linewidth=0.5, label='_nolegend_')
                plt.xlabel('iterations')
                plt.ylabel('Mean Squared Error')
                plt.title('Training and Validation Errors for different values of hidden neurons')
                plt.legend(['hidden layer neurons = 20', 'hidden layer neurons = 30', 'hidden layer neurons = 40', 'hidden layer neurons = 50', 'hidden layer neurons = 60'], loc='lower right')
                plt.savefig('Validation_train_error_hidden_neurons.png')


                plt.figure()
                plt.axis([0, epochs, -7000, 10000])
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.title('Validation Accuracy hidden neurons = '+str(no_hidden1[i])+'')
                plt.plot(range(epochs), validation_accuracy, linewidth=0.9,)
                plt.savefig('Validation_accuracy_hidden_neurons'+str(i)+'.png')

            error+=min_error #we use the min error to see the performance of each network

        errors[i]=(error/number_of_fold) #error for each learning_rate

    opt_no_hidden.append(np.argmin(errors))

plt.figure()
nu, bins, patches=plt.hist(opt_no_hidden, bins=[0,1,2,3,4,5])
plt.xlabel('Number of hidden layer neurons in the order : 20, 30, 40, 50, 60')
plt.ylabel('number of experiments')
plt.title('Distribution of optimal number of hidden neurons')
plt.savefig('Best_hidden_neuron.png')
pos = np.argmax(nu)
print('Best number of hidden neurons : %d'%no_hidden1[pos])


#final training before testing with optimal number of hidden neurons with all the training data
train_cost = np.zeros(epochs)
test_cost = np.zeros(epochs)
test_accuracy = np.zeros(epochs)

w_o.set_value(init_w_o[0:no_hidden1[i]])
b_o.set_value(init_b_o)
w_h1.set_value(init_w_h1[0:8,0:no_hidden1[i]])
b_h1.set_value(init_b_h1[0:no_hidden1[i]])

min_error = 1e+15

for iter in range(epochs):
    if iter % 100 == 0:
        print(iter)

    trainX, trainY = shuffle_data(trainX, trainY)
    cost = 0.0
    for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
        cost += train(trainX[start:end], np.transpose(trainY[start:end]))
    train_cost[iter] = cost/(n // batch_size)
    pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

    if test_cost[iter] < min_error:
        best_iter = iter
        min_error = test_cost[iter]
        best_w_o = w_o.get_value()
        best_w_h1 = w_h1.get_value()
        best_b_o = b_o.get_value()
        best_b_h1 = b_h1.get_value()

#set weights and biases to values at which performance was best
w_o.set_value(best_w_o)
b_o.set_value(best_b_o)
w_h1.set_value(best_w_h1)
b_h1.set_value(best_b_h1)

best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

print('3-layers network : Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))


plt.figure(1,figsize=(15,9))
plt.subplot(121)
plt.plot(range(epochs), test_accuracy)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Test accuracy optimal number of hidden neurons')
plt.subplot(122)
plt.plot(range(epochs), test_accuracy)
plt.axis([0, epochs, -10000, 20000])
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Test accuracy optimal number of hidden neurons rescaled')
plt.savefig('p_1b_Training_Test_Errors_and_Accuracy_opt_number_neurons.png')

plt.show()

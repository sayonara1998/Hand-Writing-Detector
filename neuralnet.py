import numpy as np
import sys
'''
quora, jpmorgan, goldman, pure storage, sig,salesforce,
Input: 
    filepath = path of the data file 
Output:
    labels = numpy array of formatted labels of data file
'''
def data_labels(filepath):
    contents = []
    count = 0
    with open(filepath) as data:
        contents = [k.strip('\n') for k in data.readlines()]
    labels = np.zeros((len(contents), 10))
    for k in contents:
        x = k.split(',')
        labels[count][int(x[0])] = 1
        count += 1
    return labels

'''
Input: 
    filepath = path of the data file 
Output:
    features = numpy array of formatted features of data file
'''
def data_features(filepath):
    contents = []
    count = 0
    with open(filepath) as data:
        contents = [k.strip('\n') for k in data.readlines()]
    features = np.zeros((len(contents), 129))
    test = []
    for k in contents:
        x = k.split(',')
        test = x[1:]
        for i in range(len(test)):
            test[i] = float(test[i])
        features[count][0] = 1
        features[count][1:] = test
        count += 1
    return features

'''
Input: 
    features = data_features(filepath)
    h_units = # of hidden units in hidden layer of NN
    flag = {1 = random from -0.1 to 0.1
            0 = all zeroes}
Output:
    alpha = matrix of set initial alpha values based on flag
    beta = matrix of set initial beta values based on flag
'''
def init_alpha_beta(features, h_units,flag):
    M = features.shape[1] #in theory its M+1 due to bias term embedded
    D = h_units #number of hidden units in hidden layer
    K = 10 #number of classes our labels can take off
    if(flag == 1):
        alphaM = np.random.uniform(-0.1, 0.1, (D,M))
        alphaM[:,0] = 0
        betaM = np.random.uniform(-0.1, 0.1, (K,D+1))
        betaM[:,0] = 0
    elif(flag == 2):
        alphaM = np.zeros((D,M))
        betaM = np.zeros((K,D+1))
    return alphaM, betaM

#Forward Propagation
'''
Input:
    x = input datapoint
    weightsM = alpha or beta matrix
Output:
    a = activation matrix 
'''
def linear_forward(x,weightsM):
    return np.matmul(x,weightsM.T)

'''
Input:
    a = activation matrix
Output:
    z = link function matrix
'''
def sigmoid_forward(a):
    return 1 / (1 + np.exp(-a))

'''
Input:
    b = beta activation matrix
Output:
    y_predicted = predicted y values
'''
def softmax_forward(b):
    return (np.exp(b))/(np.sum(np.exp(b)))

'''
Input:
    y = actual label
    y_predicted = predicted label
Output:
    J = loss
'''
def cross_entropy_forward(y,y_predicted):
    return -np.matmul((np.log(y_predicted)).T,y)

def NNForward(data_x,labels_y,alphaM,betaM):
    a = linear_forward(data_x, alphaM)
    z = sigmoid_forward(a)
    z = np.append(1, z)
    b = linear_forward(z, betaM)
    y_predicted = softmax_forward(b)
    J = cross_entropy_forward(labels_y, y_predicted)
    return data_x, a, z, b, y_predicted, J

#Backward Propagation
def linear_backward_beta(z,g_b,betaM):
    beta_star = betaM[:,1:]
    g_beta = np.outer(g_b,z)
    g_z = np.matmul((beta_star).T,g_b)
    return g_beta,g_z

def linear_backward_alpha(x,g_a):
    g_alpha = np.outer(g_a,x)
    return g_alpha

def sigmoid_backward(z,g_z):
    z_star = np.delete(z,0)
    g_a = np.multiply(g_z,(np.subtract(z_star,(np.multiply(z_star,z_star)))))
    return g_a

def softmax_backward(y,y_predicted):
    g_b = np.subtract(y_predicted,y)
    return g_b

def NNBackward(x,y,alphaM,betaM,args):
    x, a, z, b, y_predicted, J = args
    g_b = softmax_backward(y, y_predicted)
    g_beta, g_z = linear_backward_beta(z, g_b, betaM)
    g_a = sigmoid_backward(z, g_z)
    g_alpha = linear_backward_alpha(x,g_a)
    return g_alpha,g_beta

def train(features,labels,features_test,labels_test,alphaM,betaM,epochs,learning_rate):
    train_entropy = 0
    mean_train = []
    test_entropy = 0
    mean_test = []
    N = len(features)
    U = len(features_test)
    for e in range(epochs):
        for n in range(N):
            x = features[n]
            y = labels[n]
            x, a, z, b, y_predicted, J = NNForward(x, y,alphaM,betaM)
            g_alpha, g_beta = NNBackward(x, y, alphaM, betaM,[x, a, z, b, y_predicted, J])
            alphaM = alphaM - learning_rate * g_alpha
            betaM = betaM - learning_rate * g_beta


        for h in range(N):
            x_train = features[h]
            y_train = labels[h]
            x, a, z, b, y_predicted, J = NNForward(x_train,y_train,alphaM,betaM)
            train_entropy += J

        for i in range(U):
            x_test = features_test[i]
            y_test = labels_test[i]
            x, a, z, b, y_predicted, J = NNForward(x_test,y_test,alphaM,betaM)
            test_entropy += J

        mean_train.append(train_entropy/N)
        mean_test.append(test_entropy/U)

        train_entropy = 0
        test_entropy = 0

    return alphaM,betaM,mean_train,mean_test

def predict(features,labels,alphaM,betaM):
    output= []
    for i in range(len(features)):
        x = features[i]
        y = labels[i]
        x, a, z, b, y_predicted, J = NNForward(x, y,alphaM,betaM)
        output.append(np.argmax(y_predicted))
    return output

def get_error(labels_train,output_train,labels_test,output_test):
    train_size = len(labels_train)
    train_labels = []
    for i in range(train_size):
        train_labels.append(np.argmax(labels_train[i]))
    error_train = 0
    for x in range(train_size):
        if(train_labels[x] != output_train[x]):
            error_train += 1
    error_train = error_train/train_size

    test_size = len(labels_test)
    test_labels = []
    for h in range(test_size):
        test_labels.append(np.argmax(labels_test[h]))
    error_test = 0
    for y in range(test_size):
        if(test_labels[y] != output_test[y]):
            error_test += 1
    error_test = error_test/test_size

    return error_train,error_test

if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    epochs = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    train_features = data_features(train_input)
    train_labels = data_labels(train_input)

    test_features = data_features(test_input)
    test_labels = data_labels(test_input)

    alphaM, betaM = init_alpha_beta(train_features, hidden_units, flag)

    newa, newb, mean_train, mean_test = train(train_features, train_labels, test_features, test_labels, alphaM, betaM, epochs,
                                              learning_rate)

    results_train = predict(train_features, train_labels, newa, newb)
    results_test = predict(test_features, test_labels, newa, newb)

    train_error,test_error = get_error(train_labels,results_train,test_labels,results_test)

    with open(train_out, 'w') as training:
        for i in results_train:
            training.write("%s\n" % str(i))

    with open(test_out, 'w') as testing:
        for i in results_test:
            testing.write("%s\n" % str(i))

    with open(metrics_out, 'w') as metric:
        for i in range(epochs):
            metric.write("%s\n" % str("epoch=" + str(i+1) + " crossentropy(train): " +str(mean_train[i])))
            metric.write("%s\n" % str("epoch=" + str(i+1) + " crossentropy(test): " + str(mean_test[i])))
        metric.write("%s\n" % str("error(train): " + str(train_error)))
        metric.write("%s\n" % str("error(test): " + str(test_error)))



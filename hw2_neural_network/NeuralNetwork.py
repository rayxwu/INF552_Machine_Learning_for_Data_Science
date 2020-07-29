import pandas as pd
import numpy as np
from utils import *
import sys
import os


# Split train to train_x and train_y
def get_train(train, classes):
    train_x = train[:, :-1]
    train_y = label_binarize(train[:, -1], classes)
    return train_x, train_y


# Get a batch from train_x and train_y
def get_batch(train_x, train_y, batch_size=2500, min_batch_size=200):
    index = np.random.permutation(len(train_x))
    train_x, train_y = train_x[index], train_y[index]
    for batch_slice in gen_batches(train_x.shape[0], batch_size=batch_size, min_batch_size=min_batch_size):
        yield train_x[batch_slice], train_y[batch_slice]


class NeuralNetwork:
    def __init__(self, layer_node_nums):
        self.w0 = self._init_weight_bias(size=(layer_node_nums[1], layer_node_nums[0]))
        self.b0 = self._init_weight_bias(size=(layer_node_nums[1], 1))
        self.w1 = self._init_weight_bias(size=(layer_node_nums[2], layer_node_nums[1]))
        self.b1 = self._init_weight_bias(size=(layer_node_nums[2], 1))
        self.w2 = self._init_weight_bias(size=(layer_node_nums[3], layer_node_nums[2]))
        self.b2 = self._init_weight_bias(size=(layer_node_nums[3], 1))
        self.params = [self.w2, self.b2, self.w1, self.b1, self.w0, self.b0]
        self.grads = [np.zeros_like(param) for param in self.params]
        self.optimizer = AdamOptimizer(params=self.params)
    
    def _init_weight_bias(self, size):
        return np.random.uniform(size=size, low=-1, high=1)*np.sqrt(6/(size[0]+size[1]))
    
    def forword_processing(self, x):
        self._z0 = x

        # Input Layer -> Hidden Layer 1
        _y = self.w0.dot(self._z0.T)+self.b0
        self._z1 = relu(_y.T)

        # Hidden Layer 1 -> Hidden Layer 2
        _y = self.w1.dot(self._z1.T)+self.b1
        self._z2 = relu(_y.T)

        # Hidden Layer 2 -> Output Layer
        _y = self.w2.dot(self._z2.T)+self.b2
        return softmax(_y.T)

    # Use forword_processing to predict input data
    def predict(self, x):
        return self.forword_processing(x)

    # Follow the basic structure of neural network to compute output for each layer
    def forward(self, x, y):
        self._y = self.forword_processing(x)
        self.y = y
        return self._y

    # Use backward propagation to compute the grads of the hidden weights
    def backward(self):

        # Output Layer -> Hidden Layer 2
        _loss = (self._y-self.y)/self._y.shape[0]
        _loss_w2 = _loss.T.dot(self._z2)
        _loss_b2 = _loss.sum(axis=0).reshape(self.b2.shape)

        # Hidden Layer 2 -> Hidden Layer 1
        _loss = _loss.dot(self.w2)*(self._z2 != 0)
        _loss_w1 = _loss.T.dot(self._z1)
        _loss_b1 = _loss.sum(axis=0).reshape(self.b1.shape)

        # Hidden Layer 1 -> Input Layer
        _loss = _loss.dot(self.w1)*(self._z1 != 0)
        _loss_w0 = _loss.T.dot(self._z0)
        _loss_b0 = _loss.sum(axis=0).reshape(self.b0.shape)

        self.grads = [_loss_w2, _loss_b2, _loss_w1, _loss_b1, _loss_w0, _loss_b0]

    # Use the AdamOptimizer to update parameters
    def update_params(self):
        self.optimizer.update_params(self.grads)


def output(test_Y, path):
    test_Y = test_Y.argmax(axis=1)
    test_Y = list(map(lambda x: str(x), test_Y))
    test_Y = '\n'.join(test_Y)
    with open(path, 'w') as fp:
        fp.write(test_Y)


def main():
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]

    train_df = pd.read_csv(train_data_path, header=None)
    test_df = pd.read_csv(test_data_path, header=None)

    train = train_df.values
    test_X = test_df.values

    classes = sorted(list(set(train[:, -1])))
    train_X, train_Y = get_train(train, classes=classes)

    # Initializing the neural network class
    model = NeuralNetwork([3, 8, 16, len(classes)])
    accuracy_train = 0
    for i in range(Epochs):
        t_accuracy = []
        for train_x, train_y in get_batch(train_X, train_Y):
            _y = model.forward(train_x, train_y)
            t_accuracy.append(accuracy_score(_y.argmax(axis=1), train_y.argmax(axis=1)))
            model.backward()
            model.update_params()
        if accuracy_train < np.mean(t_accuracy):
            accuracy_train = np.mean(t_accuracy)
            test_Y = model.predict(test_X)

    output_name = os.path.basename(train_data_path).split("_")[0] + '_predictions.csv'
    output(test_Y, output_name)


Epochs = 650

if __name__ == '__main__':
    np.random.seed(9999)
    main()


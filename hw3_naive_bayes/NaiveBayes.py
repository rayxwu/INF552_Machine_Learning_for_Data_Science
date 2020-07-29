import sys
import numpy as np


# Get x, y from blackbox
def get_data(blackbox):
    x, y = blackbox.ask()
    x = np.array(x).reshape(-1, len(x))
    y = np.array(y).reshape(-1)
    return x, y


# Build a batch as train and test dataset
def get_batch(blackbox, batch_size=1):
    X = []
    Y = []
    for i in range(batch_size):
        x, y = get_data(blackbox)
        X.append(x)
        Y.append(y)
    X = np.array(X).reshape(-1,x.shape[1])
    Y = np.array(Y).reshape(-1)
    return X, Y


class NaiveBayes:
    def __init__(self):
        self.class_ = None
        self.mean_ = None
        self.var_ = None
        self.old_num = None
        self.epsilon_ = 1e-8

    # Update mean, variance of each feature per class
    def updating(self, x, old_mean, old_var, old_num):
        if x.shape[0] == 0:
            return old_mean, old_var, old_num

        new_num = old_num + x.shape[0]
        new_mean = (old_mean * old_num + x.sum(axis=0)) / new_num
        new_var = (old_num * old_var + old_num * (old_mean - new_mean) ** 2 + ((x - new_mean) ** 2).sum(axis=0)) / new_num
        return new_mean, new_var, new_num

    def training(self, X, Y, classes=[0, 1, 2]):
        if self.old_num is None:
            self.class_ = classes
            self.y_ = np.zeros(len(classes))
            self.mean_ = np.zeros((len(classes), X.shape[1]))
            self.var_ = np.zeros((len(classes), X.shape[1]))
            self.old_num = np.zeros(len(classes))

        for i in self.class_:
            self.y_[i] = (self.y_[i]*self.old_num[i]+(Y == i).sum())/(self.old_num[i]+X.shape[0])

            nm, nv, nn = self.updating(X[Y == i], self.mean_[i, :], self.var_[i, :], self.old_num[i])
            self.mean_[i, :] = nm
            self.var_[i, :] = nv+self.epsilon_
            self.old_num[i] = nn  

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        return self.class_[self._predict_prob(x).argmax()]

    # P(C|X) = P(C)*P(X|C) = logP(C)+logP(X|C)
    def _predict_prob(self, x):
        prob = []
        for i in self.class_:
            prior = np.log(self.y_[i]+self.epsilon_)
            likelihood = - 0.5 * np.sum(np.log(2. * np.pi * self.var_[i, :]))
            likelihood -= 0.5 * (((x - self.mean_[i, :]) ** 2)/(self.var_[i, :])).sum()
            prob.append(prior + likelihood)
        return np.array(prob)


if __name__ == '__main__':
    if sys.argv[-1] == 'blackbox31':
        from Blackbox31 import blackbox31 as bb
    elif sys.argv[-1] == 'blackbox32':
        from Blackbox32 import blackbox32 as bb

    # Hold out 200 examples as test data
    test_X, test_Y = get_batch(bb, batch_size=200)

    # Initializing Naive Bayes model
    nb = NaiveBayes()
    test_accuracy = []

    # Training Naive Bayes model
    for i in range(1001):
        batch_size = 1
        train_X, train_Y = get_batch(bb, batch_size=batch_size)
        nb.training(train_X, train_Y, classes=[0, 1, 2])
        
        # Recording test accuracy states per 10 samples
        if i % 10 == 0:
            test_accuracy.append((test_Y == nb.predict(test_X)).sum() / 200)

    # Output accuracy states
    out = []
    for i, a in enumerate(test_accuracy):
        out.append(str(i*10)+', '+str(a))
    out = out[1:]

    with open('results_'+sys.argv[-1]+'.txt', 'w') as fp:
        fp.write('\n'.join(out))

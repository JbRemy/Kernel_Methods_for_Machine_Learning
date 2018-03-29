'''
Implementation SVM
'''
from scipy import optimize
import numpy as np


class svm(object):

    def __init__(self):

        return None

    def fit(self, X_train, y_train, reg, kernel='polynomial', l=1):

        self.kernel = kernel
        self.l = l

        n = np.shape(X_train)[0]
        X = np.matrix(X_train)
        labs = np.matrix(y_train).reshape((n, 1))

        if self.kernel == 'polynomial':
            K = np.square(np.matmul(X, X.transpose()))/self.l

        if self.kernel == 'exponential':
            K = np.exp(-self.l*np.matmul(X, X.transpose()))



        cons = []
        for i in range(n):
            cons.append({'type': 'ineq',
                         'fun': cons_1,
                         'jac': jac_1,
                         'args': [labs, i, reg, n]})
            cons.append({'type': 'ineq',
                         'fun': cons_2,
                         'jac': jac_2,
                         'args': [labs, i, reg, n]})

        res = optimize.minimize(f_objective,
                                np.zeros((n,)),
                                args=(labs, K, n),
                                jac=jac_objective,
                                constraints=cons)

        print(res.message)

        self.ind = res.x != 0
        self.alpha = res.x[self.ind]

        self.X_train = X[self.ind, :]

        return None


    def predict(self, X):

        if self.kernel == 'polynomial':
            K = np.square(np.matmul(self.X_train, X.transpose()))/self.l

        if self.kernel == 'exponential':
            K = np.exp(-self.l*np.matmul(self.X_train, X.transpose()))

        scores = np.array(self.alpha.dot(K)).flatten()
        labels = []
        for _ in scores:
            if _ >= 0:
                labels.append(1)
            else:
                labels.append(-1)


        return labels, scores


def f_objective(x, y, A, n):
    _x = np.matrix(x).reshape((n, 1))

    return (-2 * np.matmul(_x.transpose(), y) + np.matmul(_x.transpose(), np.matmul(A, _x))).item()


def jac_objective(x, y, A, n):
    _x = np.matrix(x).reshape((n, 1))

    return np.array(-2 * y + 2 * np.matmul(A, _x)).flatten()


def cons_1(x, y, i, reg, n):

    return (1 / (2 * reg * n) - x[i] * y[i]).item()


def jac_1(x, y, i, reg, n):
    jac = np.zeros((n,))
    jac[i] = - y[i]

    return jac


def cons_2(x, y, i, reg, n):

    return (x[i] * y[i]).item()


def jac_2(x, y, i, reg, n):
    jac = np.zeros((n,))
    jac[i] = y[i]

    return jac

if __name__ == '__main__':

    from sklearn.datasets.samples_generator import make_blobs, make_moons
    from matplotlib import pyplot
    from pandas import DataFrame

    # X, y = make_blobs(n_samples=100, centers=2, n_features=2)
    X, y = make_moons(n_samples=100, noise=0.5)

    for _ in range(len(y)):
        if y[_] == 0:
            y[_] = -1

    df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {-1: 'red', 1: 'blue', 2: 'green'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()

    SVM = svm()
    SVM.fit(X[0:80, :], y[0:80], 0.1)
    y_pred, scores = SVM.predict(X[80:100, :])

    df = DataFrame(dict(x=X[80:100, 0], y=X[80:100, 1], label=y_pred))
    colors = {-1: 'red', 1: 'blue', 2: 'green'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
import csv
import numpy as np
import matplotlib.pyplot as plt
def Csv(filename):
    with open(filename, "r") as csvfile:
        lines = csv.reader(csvfile)
        given_set = list(lines)
        for i in range(len(given_set)):
            given_set[i] = [float(x) for x in given_set[i]]
    return np.array(given_set)
def normalize(X):
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    norm_X = 1 - ((maxs - X) / rng)
    return norm_X
def sigmond(theta, X):
    return 1.0 / (1 + np.exp(-np.dot(X, theta.T)))
def logGradient(theta, X, y):
    calc_first = sigmond(theta, X) - y.reshape(X.shape[0], -1)
    calc_final = np.dot(calc_first.T, X)
    return calc_final
def cost_func(theta, X, y):
    log_func_v = sigmond(theta, X)
    y = np.squeeze(y)
    s1 = y * np.log(log_func_v)
    s2 = (1 - y) * np.log(1 - log_func_v)
    final = -s1 - s2
    return np.mean(final)
def gradient_desc(X, y, theta, lr=.01, converge_fact=.001):
    cost = cost_func(theta, X, y)
    change_cost = 1
    num_iter = 1
    while (change_cost > converge_fact):
        old_cost = cost
        theta = theta - (lr * logGradient(theta, X, y))
        cost = cost_func(theta, X, y)
        change_cost = old_cost - cost
        num_iter += 1
    return theta, num_iter
def pred_values(theta, X):
    pred_prob = sigmond(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    return np.squeeze(pred_value)
def plot_reg(X, y, theta):
    x_0 = X[np.where(y == 0.0)]
    x_1 = X[np.where(y == 1.0)]
    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0')
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')
    x1 = np.arange(0, 1, 0.1)
    x2 = -(theta[0, 0] + theta[0, 1] * x1) / theta[0, 2]
    plt.plot(x1, x2, c='k', label='reg line')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    given_set = Csv('given_set1.csv')
    X = normalize(given_set[:, :-1])
    X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
    y = given_set[:, -1]
    theta = np.matrix(np.zeros(X.shape[1]))
    theta, num_iter = gradient_desc(X, y, theta)
    print("regression coefficients:", theta)
    print("Total iterations:", num_iter)
    y_pred = pred_values(theta, X)
    print("Correctly predicted labels:", np.sum(y == y_pred))
    plot_reg(X, y, theta)

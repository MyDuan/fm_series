from random import normalvariate
import numpy as np
import matplotlib.pyplot as plt
import math

loss_list = []

def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def train(x_train, y_train, k, epochs, train_type):
    m, n = np.shape(x_train)
    alpha = 0.01
    lamuda = 0.01
    w_0 = 0.
    w = np.zeros((n, 1))
    v = normalvariate(0, 0.1) * np.ones((n, k))
    for iter in range(epochs):
        for x in range(m):
            temp_1 = np.dot(x_train[x], v)
            temp_2 = np.dot(np.multiply(x_train[x], x_train[x]), np.multiply(v, v))
            interaction = np.sum(np.multiply(temp_1, temp_1) - temp_2) / 2.
            y = w_0 + np.dot(x_train[x], w) + interaction

            if train_type == 'regr':
                loss = (y[0] - y_train[x]) * (y[0] - y_train[x])
                delta = 2 * (y[0] - y_train[x])
            elif train_type == '2_class':
                loss = -math.log(sigmoid(y_train[x] * y[0]))
                delta = y_train[x] * (sigmoid(y_train[x] * y[0]) - 1.0)
            else:
                print("Not support this type")
                exit()

            w_0 -= alpha * (delta + 2 * lamuda * w_0)
            for i in range(n):
                w[i, 0] -= alpha * (delta * x_train[x, i] + 2 * lamuda * w[i, 0])
                for j in range(k):
                    h = x_train[x, i] * (np.dot(x_train[x], v[:, j]) - x_train[x, i] * v[i, j])
                    v[i, j] -= alpha * (delta * h + 2 * lamuda * v[i, j])

        loss_list.append(loss)
        if iter % 10 == 0:
            print(str(iter) + "_times:")
            print("Loss: ", loss)
    return w_0, w, v


def predict(w_0, w, v, x_test, train_type):
    m, n = np.shape(x_test)
    y_test = []
    for x in range(m):
        temp_1 = np.dot(x_test[x], v)
        temp_2 = np.dot(np.multiply(x_test[x], x_test[x]), np.multiply(v, v))
        interaction = np.sum(np.multiply(temp_1, temp_1) - temp_2) / 2.
        y = w_0 + np.dot(x_test[x], w) + interaction

        if train_type == 'regr':
            y_test.append(y[0])
        elif train_type == '2_class':
            if y[0] > 0:
                y_test.append(1)
            else:
                y_test.append(-1)
    return y_test


def show_result(train_type, x_train_data, y_train_data, y_results):
    if train_type == 'regr':
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 4))
        ax1.plot(x_train_data, y_results, color='red', )
        ax1.plot(x_train_data, y_train_data, color='blue')
        ax2.set_ylabel('loss')
        ax2.plot(loss_list)
        plt.show()
    elif train_type == '2_class':
        acc = 0
        for i in range(len(y_train_data)):
            if y_train_data[i] == y_results[i]:
                acc += 1
        print("accuracy:", acc / 100.0)
        print(y_results)
        plt.plot(loss_list)
        plt.show()

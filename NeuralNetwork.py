# -*- Coding = UTF-8 -*-
# @Time: 2022/3/31 16:19
# @Author: Nico
# File: NeuralNetwork.py
# @Software: PyCharm


import numpy as np


def sigmoid(x, deriv=False):
    if deriv == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


x = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1],
              [0, 0, 1]]
             )
# print(x.shape)

y = np.array([[0],
              [1],
              [1],
              [0],
              [0]]
             )
# print(y.shape)

np.random.seed(1)

w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1

# print(w0)

for j in range(50000):
    l0 = x
    l1 = sigmoid(np.dot(l0, w0))
    l2 = sigmoid(np.dot(l1, w1))
    l2_error = y - l2
    # print(l2_error.shape)
    if (j % 10000) == 0:
        print('Error' + str(np.mean(np.abs(l2_error))))
    l2_delta = l2_error * sigmoid(l2, deriv=True)
    # print(l2_delta.shape)
    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid(l1, deriv=True)

    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)


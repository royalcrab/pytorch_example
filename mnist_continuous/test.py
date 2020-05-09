from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn


a = torch.randn([1, 2])
b = torch.randn([1, 2])
mse = nn.MSELoss()


def measure_with_api(cnt):
    times = []
    since = time()
    for i in range(cnt):
        mse(a, b)
        times.append(time() - since)
    return times


def measure_without_api(cnt):
    times = []
    since = time()
    for i in range(cnt):
        torch.mean(torch.pow(torch.sub(a, b), 2))
        times.append(time() - since)
    return times


if __name__ == '__main__':
    sns.set()
    times_without = measure_without_api(1000)
    times_with = measure_with_api(1000)
    plt.plot(times_without, label="without")
    plt.plot(times_with, label="with")
    plt.title("measure")
    plt.legend()
    plt.show()

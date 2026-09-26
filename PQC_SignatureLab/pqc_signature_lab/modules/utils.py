
import time
import matplotlib.pyplot as plt

def timer(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def plot_bar(labels, values, title, path):
    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Value")
    plt.savefig(path)
    plt.close()

def plot_line(labels, values, title, path):
    plt.figure(figsize=(6,4))
    plt.plot(labels, values, marker='o')
    plt.title(title)
    plt.ylabel("Value")
    plt.savefig(path)
    plt.close()

import numpy as np
import numpy as np


def run():
    total = 0
    x = np.ascontiguousarray(np.zeros(20000))
    for i in range(200):
        total += x.sum()
    return total

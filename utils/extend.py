import numpy as np

def np_extend(a, b, axis=0):
    if a is None:
        shape = list(b.shape)
        shape[axis] = 0
        a = np.array([]).reshape(tuple(shape))
    return np.append(a, b, axis)

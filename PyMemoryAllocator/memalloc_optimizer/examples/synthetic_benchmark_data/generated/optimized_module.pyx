
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

# Arena allocator for memoryviews
cdef class Arena:
    cdef double[:] buffer
    cdef Py_ssize_t size
    cdef Py_ssize_t offset

    def __init__(self, int n):
        self.buffer = cnp.zeros(n, dtype=cnp.float64)
        self.size = n
        self.offset = 0

    cdef double[:] alloc(self, int n):
        if self.offset + n > self.size:
            raise MemoryError("Arena overflow")
        view = self.buffer[self.offset:self.offset+n]
        self.offset += n
        return view

# Example optimized function (placeholder)
def optimized_kernel(double[:] data):
    cdef Py_ssize_t i
    cdef double acc = 0
    for i in range(data.shape[0]):
        acc += data[i] * 0.5
    return acc

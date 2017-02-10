import numpy as np
cimport numpy as np

cdef extern void suffixsort(int *x, int *p, int n, int k, int l)

def suffix_sort(int[:] x, int x_min, int x_max):
    cdef int num_elements = len(x)
    cdef np.ndarray[int, ndim=1] suffix_array = np.empty(num_elements + 1, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] input_or_inverse = np.empty(num_elements + 1, dtype=np.int32)
    input_or_inverse[:num_elements] = x
    suffixsort(x=&input_or_inverse[0], p=&suffix_array[0], n=num_elements, k=x_max + 1, l=x_min)
    return suffix_array, input_or_inverse

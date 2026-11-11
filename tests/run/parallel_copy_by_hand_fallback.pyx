# mode: run
# tag: openmp, threads

# In a parallel block, a variable copied by hand is private to the whole block
# (see parallel_copy_by_hand.pyx). Code after the loop in the block that reaches
# the original variable another way, through a pointer or as a global, would see
# the value from before the block. Such variables keep firstprivate and
# lastprivate, and the doctests check the values.

# cython: test_assert_c_code_has = lastprivate\(__pyx_v_address_unused\)
# cython: test_assert_c_code_has = lastprivate\(__pyx_v_mixed_x\)
# cython: test_fail_if_c_code_has = lastprivate\(__pyx_v_mixed_y\)

cimport cython
from cython.parallel import prange, parallel

cdef enum:
    N = 100


def block_pointer_read():
    """
    >>> block_pointer_read()
    (99, 0)
    """
    cdef int i, x = -1, n_wrong = 0
    cdef int *p = &x
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            x = i
        if p[0] != N - 1:
            n_wrong_ptr[0] = 1
    return x, n_wrong


def block_pointer_write():
    """A write through the pointer after the loop is the last one.

    >>> block_pointer_write()
    100
    """
    cdef int i, x = -1
    cdef int *p = &x
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            x = i
        p[0] = 100
    return x


def block_cython_address():
    """
    >>> block_cython_address()
    (99, 0)
    """
    cdef int i, x = -1, n_wrong = 0
    cdef int *p = cython.address(x)
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            x = i
        if p[0] != N - 1:
            n_wrong_ptr[0] = 1
    return x, n_wrong


cdef struct pair:
    int a
    int b


cdef inline pair make_pair(int a) noexcept nogil:
    cdef pair r
    r.a = a
    r.b = a
    return r


def block_field_pointer():
    """
    >>> block_field_pointer()
    (99, 0)
    """
    cdef int i, n_wrong = 0
    cdef pair s
    s.a = -1
    s.b = -1
    cdef int *p = &s.b
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            s = make_pair(i)
        if p[0] != N - 1:
            n_wrong_ptr[0] = 1
    return s.b, n_wrong


cdef struct with_array:
    int arr[2]


def block_array_field():
    """An array field becomes a pointer into the struct without any &.

    >>> block_array_field()
    (99, 0)
    """
    cdef int i, n_wrong = 0
    cdef with_array s, tmp
    s.arr[0] = -1
    cdef int *p = s.arr
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            tmp.arr[0] = i
            tmp.arr[1] = i
            s = tmp
        if p[0] != N - 1:
            n_wrong_ptr[0] = 1
    return s.arr[0], n_wrong


cdef int g = -1

cdef int read_g() noexcept nogil:
    return g


def block_global():
    """A global assigned in the loop, read by a function after it.

    >>> block_global()
    (99, 0)
    """
    global g
    cdef int i, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    g = -1
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            g = i
        if read_g() != N - 1:
            n_wrong_ptr[0] = 1
    return g, n_wrong


def block_address_unused():
    """The address is taken but never read: the clauses stay all the same.

    >>> block_address_unused()
    99
    """
    cdef int i, address_unused = -1
    cdef int *p = &address_unused
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            address_unused = i
    return address_unused


def block_mixed():
    """Only the variable whose address is taken keeps the clauses.

    >>> block_mixed()
    (99, 99, 0)
    """
    cdef int i, mixed_x = -1, mixed_y = -1, n_wrong = 0
    cdef int *p = &mixed_x
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            mixed_x = i
            mixed_y = i
        if p[0] != N - 1 or mixed_y != N - 1:
            n_wrong_ptr[0] = 1
    return mixed_x, mixed_y, n_wrong

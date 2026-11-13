# mode: run
# tag: openmp, threads

# A prange in a parallel() block keeps firstprivate and lastprivate: code of the
# block before and after the loop may read its variables. Only a prange outside
# of a block copies them by hand (see parallel_copy_by_hand.pyx). The doctests
# check that the values in a block don't change.

# cython: test_assert_c_code_has = firstprivate\(__pyx_v_x\) lastprivate\(__pyx_v_x\)
# cython: test_fail_if_c_code_has = __Pyx_ParallelSnapshot

cimport cython
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

cdef enum:
    N = 100


def block_read():
    """Read in the parallel block, after the loop.

    >>> block_read()
    0
    """
    cdef int block_read = -1, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for block_read in prange(N):
            pass
        if block_read != N - 1:
            with gil:
                n_wrong_ptr[0] += 1
    return n_wrong




def read_in_block_and_after():
    """Each thread sees the last value after the loop, and so does the code after the block.

    >>> read_in_block_and_after()
    (99, 0)
    """
    cdef int i, x = -1, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            x = i
        if x != N - 1:
            n_wrong_ptr[0] = 1
    return x, n_wrong


def read_before_loop():
    """Code before the loop sees the value from before the block.

    >>> read_before_loop()
    (99, 0)
    """
    cdef int i, x = 7, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        if x != 7:
            n_wrong_ptr[0] = 1
        for i in prange(N):
            x = i
    return x, n_wrong


def else_ends_block():
    """The else clause of the last loop of the block sees the last value.

    >>> else_ends_block()
    (99, 0)
    """
    cdef int i, x = -1, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            x = i
        else:
            if x != N - 1:
                n_wrong_ptr[0] = 1
    return x, n_wrong


def two_loops():
    """The second loop copies in the last value of the first one.

    >>> two_loops()
    (99, 0)
    """
    cdef int i, j, x = -1, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            if i % 2:
                x = i
        for j in prange(N):
            if x != N - 1:
                n_wrong_ptr[0] = 1
            if j < 0:
                x = j
    return x, n_wrong


def unassigned_keeps_value():
    """Assigned in no iteration: the value from before the block comes back out.

    >>> unassigned_keeps_value()
    7
    """
    cdef int i, x = 7
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            if i < 0:
                x = i
    return x


def loop_in_block_loop():
    """A prange run several times in the block: each run copies in the last value.

    >>> loop_in_block_loop()
    (299, 0)
    """
    cdef int r, i, x = -1, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for r in range(3):
            if x != r * N - 1:
                n_wrong_ptr[0] = 1
            for i in prange(N):
                x = r * N + i
    return x, n_wrong


def schedules_in_block():
    """
    >>> schedules_in_block()
    [99, 99, 99, 99]
    """
    cdef int i, a = -1, b = -1, c = -1, d = -1
    with nogil, parallel(num_threads=4):
        for i in prange(N, schedule='static', chunksize=7):
            a = i
        for i in prange(N, schedule='dynamic', chunksize=3):
            b = i
        for i in prange(N, schedule='guided'):
            c = i
        for i in prange(N, schedule='runtime'):
            d = i
    return [a, b, c, d]


def target_after_block():
    """
    >>> target_after_block()
    99
    """
    cdef int i = -1
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            pass
    return i


def per_thread_buffer():
    """The scikit-learn pattern: allocate per thread, loop, free.

    >>> per_thread_buffer()
    4950
    """
    cdef int i
    cdef long total = 0
    cdef long *total_ptr = &total
    cdef long *buf
    with nogil, parallel(num_threads=4):
        buf = <long *> malloc(sizeof(long))
        buf[0] = 0
        for i in prange(N):
            buf[0] += i
        with gil:
            total_ptr[0] += buf[0]
        free(buf)
    return total



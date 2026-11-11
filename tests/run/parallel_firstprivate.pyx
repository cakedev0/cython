# mode: run
# tag: openmp, threads

# A variable assigned in a prange is firstprivate only when a read in the loop
# body may see its value from before the loop. A variable in both firstprivate
# and lastprivate makes GCC and clang add a barrier at the start of the loop.
# The doctests check the values where firstprivate is needed, and the C code
# checks that it's left out elsewhere.

# cython: test_fail_if_c_code_has = firstprivate\(__pyx_v_target\)
# cython: test_fail_if_c_code_has = firstprivate\(__pyx_v_body_local\)
# cython: test_fail_if_c_code_has = firstprivate\(__pyx_v_nested_local\)
# cython: test_fail_if_c_code_has = firstprivate\(__pyx_v_struct_assigned\)
# cython: test_fail_if_c_code_has = firstprivate\(__pyx_v_block_local\)
# cython: test_fail_if_c_code_has = firstprivate\(__pyx_v_read_after\)
# cython: test_assert_c_code_has = firstprivate\(__pyx_v_copied_in\)
# cython: test_assert_c_code_has = firstprivate\(__pyx_v_previous_iteration\)

cimport cython
from cython.parallel import prange, parallel

cdef enum:
    N = 100


def target():
    """
    >>> target()
    (99, 4950)
    """
    cdef int target, total = 0
    for target in prange(N, nogil=True, num_threads=4):
        total += target
    return target, total


def body_local():
    """
    >>> body_local()
    9900
    """
    cdef int i, body_local, total = 0
    for i in prange(N, nogil=True, num_threads=4):
        body_local = 2 * i
        total += body_local
    return total


def nested_local():
    """
    >>> nested_local()
    49500
    """
    cdef int i, j, nested_local, total = 0
    for i in prange(N, nogil=True, num_threads=4):
        for j in range(10):
            nested_local = i
            total += nested_local
    return total


def read_after():
    """lastprivate stays: the value after the loop is the last one.

    >>> read_after()
    198
    """
    cdef int i, read_after = -1
    for i in prange(N, nogil=True, num_threads=4):
        read_after = 2 * i
    return read_after


cdef struct pair:
    int a
    int b


cdef inline pair make_pair(int a) noexcept nogil:
    cdef pair r
    r.a = a
    r.b = a
    return r


def struct_assigned():
    """
    >>> struct_assigned()
    4950
    """
    cdef int i, total = 0
    cdef pair struct_assigned
    for i in prange(N, nogil=True, num_threads=4):
        struct_assigned = make_pair(i)
        total += struct_assigned.b
    return total


def block_local():
    """
    >>> block_local()
    (99, 0)
    """
    cdef int i, block_local = -1, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            block_local = i
            if block_local != i:
                n_wrong_ptr[0] = 1
        if block_local != N - 1:
            n_wrong_ptr[0] = 1
    return block_local, n_wrong


# A read may see the value from before the loop: firstprivate stays.


def copied_in():
    """
    >>> copied_in()
    0
    """
    cdef int i, copied_in = 7, n_wrong = 0
    for i in prange(N, nogil=True, num_threads=4):
        if copied_in != 7:
            n_wrong += 1
        if i < 0:
            copied_in = 0
    return n_wrong


def previous_iteration():
    """Uninitialized before the loop, but a read may see the previous iteration.

    >>> previous_iteration()
    True
    """
    cdef int i, previous_iteration, n_seen = 0
    for i in prange(N, nogil=True, num_threads=4):
        if i > 0 and previous_iteration == i - 1:
            n_seen += 1
        previous_iteration = i
    return n_seen > 0


def conditional_assignment():
    """
    >>> conditional_assignment()
    0
    """
    cdef int i, x = 7, n_wrong = 0
    for i in prange(N, nogil=True, num_threads=4):
        if i % 2:
            x = i
        if x != 7 and x % 2 == 0:
            n_wrong += 1
    return n_wrong


def address_in_body():
    """A read through the address of the private copy.

    >>> address_in_body()
    0
    """
    cdef int i, x = 7, n_wrong = 0
    cdef int *p
    for i in prange(N, nogil=True, num_threads=4):
        p = &x
        if p[0] != 7:
            n_wrong += 1
        if i < 0:
            x = 0
    return n_wrong


def struct_field_before():
    """A field set before the loop, read before the struct is assigned.

    >>> struct_field_before()
    0
    """
    cdef int i, n_wrong = 0
    cdef pair s
    s.a = 7
    for i in prange(N, nogil=True, num_threads=4):
        if s.a != 7:
            n_wrong += 1
        if i < 0:
            s = make_pair(i)
    return n_wrong


def argument(int x):
    """
    >>> argument(7)
    0
    """
    cdef int i, n_wrong = 0
    for i in prange(N, nogil=True, num_threads=4):
        if x != 7:
            n_wrong += 1
        if i < 0:
            x = 0
    return n_wrong


def read_in_block_before_loop():
    """
    >>> read_in_block_before_loop()
    0
    """
    cdef int i, x = 7, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            if x != 7:
                n_wrong_ptr[0] = 1
            if i < 0:
                x = 0
    return n_wrong

# tag: run
# tag: openmp, threads

# A prange that holds the GIL writes its lastprivate values back by hand: clang
# adds a barrier at the end of a loop with lastprivate, even with nowait, and the
# threads would wait there with the GIL held. Each loop here keeps firstprivate,
# and the doctests check the values that lastprivate would give.

# cython: freethreading_compatible=True
# cython: test_fail_if_c_code_has = \blastprivate\(
# cython: test_assert_c_code_has = firstprivate\(__pyx_v_i\)

from cython.parallel import prange, parallel, threadid

cdef struct pair:
    int a
    int b

def target_read_after(int n):
    """
    >>> target_read_after(1000)
    (999, 1000)
    """
    cdef int i = -1
    cdef int total = 0
    for i in prange(n, num_threads=4):
        total += 1
    return i, total

def target_untouched_without_iterations():
    """
    >>> target_untouched_without_iterations()
    -1
    """
    cdef int i = -1
    for i in prange(0, num_threads=4):
        pass
    return i

def local_read_after(int n):
    """
    >>> local_read_after(1000)
    1998
    """
    cdef int i, x = -1
    for i in prange(n, num_threads=4):
        x = 2 * i
    return x

def read_before_write(int n):
    """
    >>> read_before_write(1000)
    1099
    """
    cdef int i, y = 100
    for i in prange(n, num_threads=4):
        if i == n - 1:
            y = y + i
    return y

def struct_read_after(int n):
    """
    >>> struct_read_after(1000)
    (999, 1998)
    """
    cdef int i
    cdef pair p
    p.a = p.b = -1
    for i in prange(n, num_threads=4):
        p = pair(i, 2 * i)
    return p.a, p.b

def python_calls_in_body(int n):
    """
    >>> python_calls_in_body(1000)
    (999, 999)
    """
    cdef int i, x = -1
    out = [None] * n
    for i in prange(n, num_threads=4):
        out[i] = i
        x = out[i]
    assert out == list(range(n))
    return i, x

def read_in_parallel_block(int n):
    """
    >>> read_in_parallel_block(1000)
    {999}
    """
    cdef int i, x = -1
    seen = [None] * 4
    with parallel(num_threads=4):
        for i in prange(n):
            x = i
        seen[threadid()] = x
    return {v for v in seen if v is not None}

def ends_parallel_block(int n):
    """
    >>> ends_parallel_block(1000)
    (999, 999)
    """
    cdef int i, x = -1
    with parallel(num_threads=4):
        for i in prange(n):
            x = i
    return i, x

def loop_with_break(int n):
    """
    >>> loop_with_break(1000)
    """
    cdef int i, x = -1
    for i in prange(n, num_threads=4):
        x = i
        if i == 10:
            break

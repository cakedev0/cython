# tag: run
# tag: openmp, threads

# Variables of a prange only need firstprivate + lastprivate, and so the barrier
# that comes with them on GCC and clang, if the difference can be observed.
#
# The doctests check the values: where a variable must keep both clauses, a
# wrong private leaves the variable at its value from before the loop. The C
# asserts check that the barrier goes away, and that the clauses stay where a
# wrong private can't fail reliably: reading an uninitialized value, or a break,
# where the result may legitimately be the value from before the loop.

# cython: test_assert_c_code_has = \bprivate\(__pyx_v_unused_target\)
# cython: test_assert_c_code_has = \bprivate\(__pyx_v_reused_target\)
# cython: test_assert_c_code_has = \bprivate\(__pyx_v_overwritten_target\)
# cython: test_assert_c_code_has = \bprivate\(__pyx_v_body_local\)
# cython: test_assert_c_code_has = \bprivate\(__pyx_v_nested_unused\)
# cython: test_assert_c_code_has = \bprivate\(__pyx_v_struct_whole\)
# cython: test_fail_if_c_code_has = lastprivate\(__pyx_v_unused_target\)
# cython: test_fail_if_c_code_has = lastprivate\(__pyx_v_reused_target\)
# cython: test_fail_if_c_code_has = lastprivate\(__pyx_v_overwritten_target\)
# cython: test_fail_if_c_code_has = lastprivate\(__pyx_v_body_local\)
# cython: test_fail_if_c_code_has = lastprivate\(__pyx_v_nested_unused\)
# cython: test_fail_if_c_code_has = lastprivate\(__pyx_v_struct_whole\)

# cython: test_assert_c_code_has = firstprivate\(__pyx_v_copied_in\)
# cython: test_assert_c_code_has = firstprivate\(__pyx_v_maybe_set\)
# cython: test_assert_c_code_has = firstprivate\(__pyx_v_struct_partial\)
# cython: test_assert_c_code_has = lastprivate\(__pyx_v_block_read\)
# cython: test_assert_c_code_has = lastprivate\(__pyx_v_break_read\)

cimport cython.parallel
from cython.parallel import prange
from libc.stdlib cimport malloc, free

cdef enum:
    N = 100


def unused_target():
    """
    >>> unused_target()
    4950
    """
    cdef int unused_target, total = 0
    for unused_target in prange(N, nogil=True, num_threads=4):
        total += unused_target
    return total


def reused_target():
    """Each loop doesn't see the assignments of the other one.

    >>> reused_target()
    9900
    """
    cdef int reused_target, total = 0
    for reused_target in prange(N, nogil=True, num_threads=4):
        total += reused_target
    for reused_target in prange(N, nogil=True, num_threads=4):
        total += reused_target
    return total


def overwritten_target():
    """
    >>> overwritten_target()
    -1
    """
    cdef int overwritten_target
    for overwritten_target in prange(N, nogil=True, num_threads=4):
        pass
    overwritten_target = -1
    return overwritten_target


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


def nested_unused():
    """
    >>> nested_unused()
    300
    """
    cdef int i, nested_unused, total = 0
    for i in prange(N, nogil=True, num_threads=4):
        for nested_unused in prange(3):
            total += 1
    return total


cdef struct pair:
    int a
    int b


cdef inline pair make_pair(int a) noexcept nogil:
    cdef pair p
    p.a = a
    p.b = 2 * a
    return p


def struct_whole():
    """
    >>> struct_whole()
    14850
    """
    cdef int i, total = 0
    cdef pair struct_whole
    for i in prange(N, nogil=True, num_threads=4):
        struct_whole = make_pair(i)
        total += struct_whole.a + struct_whole.b
    return total


def struct_partial():
    """Assigned in some iterations only, the value comes from before the loop.

    >>> struct_partial()
    700
    """
    cdef int i, total = 0
    cdef pair struct_partial
    struct_partial.a = 0
    struct_partial.b = 7
    for i in prange(N, nogil=True, num_threads=4):
        if i < 0:
            struct_partial = make_pair(i)
        total += struct_partial.b
    return total


def struct_read():
    """
    >>> struct_read()
    (99, 198)
    """
    cdef int i
    cdef pair struct_read
    struct_read.a = -1
    struct_read.b = -1
    for i in prange(N, nogil=True, num_threads=4):
        struct_read = make_pair(i)
    return struct_read.a, struct_read.b


def read_target(int n):
    """
    >>> read_target(100)
    99
    >>> read_target(0)
    -1
    """
    cdef int read_target = -1
    for read_target in prange(n, nogil=True, num_threads=4):
        pass
    return read_target


def read_local():
    """
    >>> read_local()
    198
    """
    cdef int i, read_local = -1
    for i in prange(N, nogil=True, num_threads=4):
        read_local = 2 * i
    return read_local


def cond_local():
    """
    >>> cond_local()
    99
    """
    cdef int i, cond_local = 7
    for i in prange(N, nogil=True, num_threads=4):
        if i % 2:
            cond_local = i
    return cond_local


def copied_in():
    """The value from before the loop is seen by every thread's first iteration.

    >>> copied_in()
    True
    """
    cdef int i, copied_in = 7
    cdef int n_seven = 0
    for i in prange(N, nogil=True, num_threads=4, schedule='static'):
        if copied_in == 7:
            n_seven += 1
        copied_in = i
    return n_seven >= 1


def maybe_set():
    """
    >>> maybe_set()
    100
    """
    cdef int i, maybe_set = 7, n_seven = 0
    for i in prange(N, nogil=True, num_threads=4):
        if i < 0:
            maybe_set = 0
        if maybe_set == 7:
            n_seven += 1
    return n_seven


def address_taken():
    """Read through a pointer: flow control can't see this one.

    >>> address_taken()
    99
    """
    cdef int i, address_taken = -1
    cdef int *ptr = &address_taken
    for i in prange(N, nogil=True, num_threads=4):
        address_taken = i
    return ptr[0]


def field_address():
    """A pointer to a field, then the whole struct is assigned in the loop.

    >>> field_address()
    99
    """
    cdef int i
    cdef pair field_address
    field_address.a = -1
    cdef int *ptr = &field_address.a
    for i in prange(N, nogil=True, num_threads=4):
        field_address = make_pair(i)
    return ptr[0]


cdef struct with_array:
    int arr[2]


def array_field():
    """An array field becomes a pointer into the struct without any &.

    >>> array_field()
    99
    """
    cdef int i
    cdef with_array array_field, tmp
    array_field.arr[0] = -1
    cdef int *ptr = array_field.arr
    for i in prange(N, nogil=True, num_threads=4):
        tmp.arr[0] = i
        tmp.arr[1] = i
        array_field = tmp
    return ptr[0]


def else_read():
    """
    >>> else_read()
    99
    """
    cdef int else_read = -1, res = 0
    for else_read in prange(N, nogil=True, num_threads=4):
        pass
    else:
        res = else_read
    return res


def break_read():
    """
    >>> 0 <= break_read() < 100
    True
    """
    cdef int break_read = 0
    for break_read in prange(N, nogil=True, num_threads=4):
        if break_read >= 0:
            break
    return break_read


def block_read():
    """Read in the parallel block, after the loop.

    >>> block_read()
    0
    """
    cdef int block_read = -1, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    with nogil, cython.parallel.parallel(num_threads=4):
        for block_read in prange(N):
            pass
        if block_read != N - 1:
            with gil:
                n_wrong_ptr[0] += 1
    return n_wrong


def outer_loop_read():
    """Read before the loop, but after it in the next iteration of the outer loop.

    >>> outer_loop_read()
    99
    """
    cdef int r, res = -1, outer_loop_read = -1
    for r in range(2):
        if r == 1:
            res = outer_loop_read
        for outer_loop_read in prange(N, nogil=True, num_threads=4):
            pass
    return res


def locals_read():
    """
    >>> locals_read()
    99
    """
    cdef int locals_read = -1
    for locals_read in prange(N, nogil=True, num_threads=4):
        pass
    return locals()['locals_read']


def nested_read():
    """
    >>> nested_read()
    2
    """
    cdef int i, nested_read = -1
    for i in prange(N, nogil=True, num_threads=4):
        for nested_read in prange(3):
            pass
    return nested_read


def closure_read():
    """
    >>> closure_read()
    99
    """
    cdef int closure_read = -1
    def inner():
        return closure_read
    for closure_read in prange(N, nogil=True, num_threads=4):
        pass
    return inner()


cdef int module_target = -1
for module_target in prange(N, nogil=True, num_threads=4):
    pass


def module_read():
    """
    >>> module_read()
    99
    """
    return module_target

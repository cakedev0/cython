# mode: run
# tag: openmp, threads

# A prange copies its variables in from before the loop and out after it by hand,
# which avoids the barrier that GCC and clang put at the start of a loop with a
# variable in both firstprivate and lastprivate. The doctests compare the values
# with a sequential run, in cases where only the copy-in, only the copy-out or
# both can be seen.

# cython: test_fail_if_c_code_has = lastprivate
# cython: test_fail_if_c_code_has = omp for[^\n]*firstprivate

cimport cython.parallel
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free

cdef enum:
    N = 100

ctypedef fused num:
    int
    double


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


# In a parallel block


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


# Schedules, continue, fused functions, generators


def fused_read(num dummy):
    """
    >>> fused_read(1)
    99.0
    >>> fused_read(1.0)
    99.0
    """
    cdef int i
    cdef num fused_read = -1
    for i in prange(N, nogil=True, num_threads=4):
        fused_read = i
    return <double>fused_read


def gen_read():
    """prange in a generator: its locals live in a closure struct.

    >>> list(gen_read())
    [99]
    """
    cdef int gen_read = -1
    for gen_read in prange(N, nogil=True, num_threads=4):
        pass
    yield gen_read


def continue_read(int n):
    """continue on the last iteration must still copy the value out.

    >>> continue_read(100)
    99
    """
    cdef int i, continue_read = -1
    for i in prange(n, nogil=True, num_threads=4):
        continue_read = i
        if i == n - 1:
            continue
        continue_read = i
    return continue_read


def copy_in_then_out(int n):
    """Assigned only in some iterations: the value from before the loop must survive.

    >>> copy_in_then_out(100)
    7
    """
    cdef int i, copy_in_then_out = 7
    for i in prange(n, nogil=True, num_threads=4):
        if i < 0:
            copy_in_then_out = i
    return copy_in_then_out


def copy_out_schedules(int n):
    """The copy-out after the loop (static) and in the loop (other schedules).

    >>> copy_out_schedules(1000)
    [999, 999, 999, 999, 999, 999]
    """
    cdef int i, x, res = 0
    out = []
    x = -1
    for i in prange(n, nogil=True, num_threads=4):
        x = i
    out.append(x)
    x = -1
    for i in prange(n, nogil=True, num_threads=4, schedule='static'):
        x = i
    out.append(x)
    x = -1
    for i in prange(n, nogil=True, num_threads=4, schedule='static', chunksize=7):
        x = i
    out.append(x)
    x = -1
    for i in prange(n, nogil=True, num_threads=4, schedule='dynamic', chunksize=3):
        x = i
    out.append(x)
    x = -1
    for i in prange(n, nogil=True, num_threads=4, schedule='guided'):
        x = i
    out.append(x)
    x = -1
    for i in prange(n, nogil=True, num_threads=4, schedule='runtime'):
        x = i
    out.append(x)
    return out

# mode: run
# tag: cpp, openmp, threads

# A C++ reference parameter keeps the address of its argument. The variable is
# copied out of the prange by hand (see parallel_copy_by_hand.pyx), and a read
# through that address after the loop must see the last value. In a parallel
# block, it keeps the clauses (see parallel_copy_by_hand_fallback.pyx).

from cython.parallel import prange, parallel
from libcpp.vector cimport vector

cdef extern from *:
    """
    static int *saved_ptr = 0;
    static void keep_ref(int &v) { saved_ptr = &v; }
    static int read_saved(void) { return *saved_ptr; }
    """
    void keep_ref(int &v) noexcept nogil
    int read_saved() noexcept nogil

cdef enum:
    N = 100


def reference_argument():
    """
    >>> reference_argument()
    99
    """
    cdef int i, reference_argument = -1
    keep_ref(reference_argument)
    for i in prange(N, nogil=True, num_threads=4):
        reference_argument = i
    return read_saved()


def reference_argument_in_block():
    """In a parallel block, the variable keeps the clauses.

    >>> reference_argument_in_block()
    (99, 0)
    """
    cdef int i, x = -1, n_wrong = 0
    cdef int *n_wrong_ptr = &n_wrong
    keep_ref(x)
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            x = i
        if read_saved() != N - 1:
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


def field_reference_in_block():
    """A reference to a field also points into the struct.

    >>> field_reference_in_block()
    (99, 0)
    """
    cdef int i, n_wrong = 0
    cdef pair s
    cdef int *n_wrong_ptr = &n_wrong
    s.a = -1
    s.b = -1
    keep_ref(s.b)
    with nogil, parallel(num_threads=4):
        for i in prange(N):
            s = make_pair(i)
        if read_saved() != N - 1:
            n_wrong_ptr[0] = 1
    return s.b, n_wrong


# Structs that memcpy can't copy keep the clauses.

cdef struct holder:
    vector[int] v
    int n


cdef holder make_holder(int n) noexcept nogil:
    cdef holder r
    r.v.resize(n % 7)
    r.n = n
    return r


def cpp_member():
    """
    >>> cpp_member()
    (99, 1, 295)
    """
    cdef int i, total = 0
    cdef holder h
    h.v.resize(1000)
    h.n = -1
    for _ in range(20):
        total = 0
        for i in prange(100, nogil=True, num_threads=4):
            h = make_holder(i)
            total += h.v.size()
    return h.n, h.v.size(), total


cdef extern from *:
    """
    #include <string>
    struct partly_declared { int n; std::string s; };
    static partly_declared make_partly(int n) {
        partly_declared r;
        r.n = n;
        r.s = std::string(100 + n % 7, 'x');
        return r;
    }
    """
    cdef struct partly_declared:
        int n
    partly_declared make_partly(int n) noexcept nogil


def extern_struct():
    """Its declaration may leave out C++ members.

    >>> extern_struct()
    99
    """
    cdef int i
    cdef partly_declared p = make_partly(-1)
    for _ in range(20):
        for i in prange(100, nogil=True, num_threads=4):
            p = make_partly(i)
    return p.n

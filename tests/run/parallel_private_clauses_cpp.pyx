# tag: run
# tag: cpp, openmp, threads

# A C++ reference parameter keeps the address of its argument, so the variable
# keeps firstprivate + lastprivate (see parallel_private_clauses.pyx).

from cython.parallel import prange

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

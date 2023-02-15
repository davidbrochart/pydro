import numpy as np


# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

np.import_array()


cimport cython


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def run_gr4j(
    np.ndarray[np.float32_t, ndim=1] x,
    np.ndarray[np.float32_t, ndim=1] p,
    np.ndarray[np.float32_t, ndim=1] e,
    np.ndarray[np.float32_t, ndim=1] q,
    np.ndarray[np.float32_t, ndim=1] s,
    np.ndarray[np.float32_t, ndim=1] uh1_array,
    np.ndarray[np.float32_t, ndim=1] uh2_array,
    np.uint32_t l,
    np.uint32_t m,
):
    cdef np.uint32_t i, t
    cdef np.float32_t pn, ps, en, tmp, perc, pr_0, pr_i, q9, q1, f, qr, qd

    for t in range(p.size):
        if p[t] > e[t]:
            pn = p[t] - e[t]
            en = 0.
            tmp = s[0] / x[0]
            ps = x[0] * (1. - tmp ** 2) * np.tanh(pn / x[0]) / (1. + tmp * np.tanh(pn / x[0]))
            s[0] += ps
        elif p[t] < e[t]:
            ps = 0.
            pn = 0.
            en = e[t] - p[t]
            tmp = s[0] / x[0]
            es = s[0] * (2. - tmp) * np.tanh(en / x[0]) / (1. + (1. - tmp) * np.tanh(en / x[0]))
            tmp = s[0] - es
            if tmp > 0.:
                s[0] = tmp
            else:
                s[0] = 0.
        else:
            pn = 0.
            en = 0.
            ps = 0.
        tmp = (4. * s[0] / (9. * x[0]))
        perc = s[0] * (1. - (1. + tmp ** 4) ** (-1. / 4.))
        s[0] -= perc
        pr_0 = perc + pn - ps
        q9 = 0.
        q1 = 0.
        for i in range(m):
            if i == 0:
                pr_i = pr_0
            else:
                pr_i = s[2 + i - 1]
            if i < l:
                q9 += uh1_array[i] * pr_i
            q1 += uh2_array[i] * pr_i
        q9 *= 0.9
        q1 *= 0.1
        f = x[1] * ((s[1] / x[2]) ** (7. / 2.))
        tmp = s[1] + q9 + f
        if tmp > 0.:
            s[1] = tmp
        else:
            s[1] = 0.
        tmp = s[1] / x[2]
        qr = s[1] * (1. - ((1. + tmp ** 4) ** (-1. / 4.)))
        s[1] -= qr
        tmp = q1 + f
        if tmp > 0.:
            qd = tmp
        else:
            qd = 0.
        q[t] = qr + qd
        if s.size > 2:
            s[3:] = s[2:s.size - 1]
            s[2] = pr_0


cdef class GR4J:

    cdef np.float32_t[:] x, s, uh1_array, uh2_array
    cdef np.uint32_t l, m

    def __cinit__(self, x):
        cdef np.uint32_t i, x3_int

        self.x = np.array(x, dtype=np.float32)
        x3_int = np.uint32(self.x[3])
        self.s = np.empty(2 + 2 * x3_int, dtype=np.float32)
        self.s[0] = self.x[0] / 2.
        self.s[1] = self.x[2] / 2.
        self.s[2:] = 0.
        self.l = x3_int + 1
        self.m = 2 * x3_int + 1
        self.uh1_array = np.empty(self.l, dtype=np.float32)
        self.uh2_array = np.empty(self.m, dtype=np.float32)
        for i in range(self.m):
            if i < self.l:
                self.uh1_array[i] = self.uh1(i + 1)
            self.uh2_array[i] = self.uh2(i + 1)

    cpdef np.float32_t sh1(self, np.uint32_t t):
        cdef np.float32_t res, t_float

        t_float = t
        if t == 0:
            res = 0.
        elif t_float < self.x[3]:
            res = (t_float / self.x[3]) ** 2.5
        else:
            res = 1.
        return res

    cdef np.float32_t sh2(self, np.uint32_t t):
        cdef np.float32_t res, t_float

        t_float = t
        if t == 0:
            res = 0.
        elif t_float < self.x[3]:
            res = 0.5 * ((t_float / self.x[3]) ** 2.5)
        elif t_float < 2. * self.x[3]:
            res = 1. - 0.5 * ((2. - t_float / self.x[3]) ** 2.5)
        else:
            res = 1.
        return res

    cpdef np.float32_t uh1(self, np.uint32_t j):
        return self.sh1(j) - self.sh1(j - 1)

    cpdef np.float32_t uh2(self, np.uint32_t j):
        return self.sh2(j) - self.sh2(j - 1)

    cpdef np.ndarray[np.float32_t, ndim=2] run(self, pe):
        cdef np.ndarray[np.float32_t, ndim=2] _pe, q

        _pe = np.vstack(pe).astype(np.float32)
        q = np.empty((1, _pe.shape[1]), dtype=np.float32)
        run_gr4j(
            np.asarray(self.x, dtype=np.float32),
            _pe[0],
            _pe[1],
            q[0],
            np.asarray(self.s, dtype=np.float32),
            np.asarray(self.uh1_array, dtype=np.float32),
            np.asarray(self.uh2_array, dtype=np.float32),
            self.l,
            self.m,
        )
        return q

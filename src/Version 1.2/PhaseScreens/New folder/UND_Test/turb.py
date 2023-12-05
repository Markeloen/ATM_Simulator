
import numpy
from scipy.special import gamma, kv

__all__ = ["phase_covariance"]

def phase_covariance(r, r0, L0):
    # EQ 5
    r = numpy.float32(r)
    r0 = float(r0)
    L0 = float(L0)

    # Get rid of any zeros
    r += 1e-40

    A = (L0 / r0) ** (5. / 3)

    B1 = (2 ** (-5. / 6)) * gamma(11. / 6) / (numpy.pi ** (8. / 3))
    B2 = ((24. / 5) * gamma(6. / 5)) ** (5. / 6)

    C = (((2 * numpy.pi * r) / L0) ** (5. / 6)) * kv(5. / 6, (2 * numpy.pi * r) / L0)

    cov = A * B1 * B2 * C

    return cov
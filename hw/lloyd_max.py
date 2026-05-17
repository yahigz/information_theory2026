import math
import numpy as np

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def phi(x):
    return math.exp( - 0.5 * x * x) / SQRT2PI

def Phi(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def truncated_moments(a, b):
    """Return (p, m1, m2) where
    p = P(a<X<b), m1 = ∫_a^b x f(x) dx, m2 = ∫_a^b x^2 f(x) dx for standard normal.
    Analytical formulas used to avoid numeric integration.
    """
    pa = Phi(a)
    pb = Phi(b)
    p = pb - pa
    if p == 0.0:
        return 0.0, 0.0, 0.0

    phi_a = phi(a) if math.isfinite(a) else 0.0
    phi_b = phi(b) if math.isfinite(b) else 0.0
    term_a = a * phi_a if math.isfinite(a) else 0.0
    term_b = b * phi_b if math.isfinite(b) else 0.0

    m1 = phi_a - phi_b
    m2 = p - (term_b - term_a)
    return p, m1, m2

def lloyd_max(M, tol=1e - 12, max_iter=1000):
    A = 4.2
    ys = np.linspace( - A, A, M).astype(float)

    for it in range(max_iter):
        rs = [ - math.inf]
        for i in range(1, M):
            rs.append(0.5 * (ys[i - 1] + ys[i]))
        rs.append(math.inf)

        new_ys = ys.copy()
        for i in range(M):
            a = rs[i]
            b = rs[i + 1]
            p, m1, m2 = truncated_moments(a, b)
            if p > 0:
                new_ys[i] = m1 / p
            else:
                new_ys[i] = ys[i]

        diff = np.max(np.abs(new_ys - ys))
        ys = new_ys
        if diff < tol:
            break

    rs = [ - math.inf]
    for i in range(1, M):
        rs.append(0.5 * (ys[i - 1] + ys[i]))
    rs.append(math.inf)

    D = 0.0
    for i in range(M):
        a = rs[i]
        b = rs[i + 1]
        p, m1, m2 = truncated_moments(a, b)
        y = ys[i]
        D  + = m2 - 2.0 * y * m1 + (y * y) * p

    return ys.tolist(), rs, D

def format_interval(a, b):
    la = " - inf" if a ==  - math.inf else f"{a:.6f}"
    lb = " + inf" if b == math.inf else f"{b:.6f}"
    return f"({la}, {lb})"

def run_all():
    Ms = [2, 4, 5, 8]
    for M in Ms:
        ys, rs, D = lloyd_max(M)
        print(f"M = {M}")
        print("Reconstruction levels (y_i):")
        for i, y in enumerate(ys):
            print(f"  y_{i + 1} = {y:.10f}")
        print("Decision intervals:")
        for i in range(M):
            print(f"  region {i + 1}: {format_interval(rs[i], rs[i + 1])}")
        print(f"MSE D = {D:.12f}\n")

if __name__ == '__main__':
    run_all()

"""Shared signal generators for held-out estimator probes (review only)."""
import numpy as np


def rk4(f, x0, dt, n, burn=5000, sub=1):
    x = np.asarray(x0, dtype=float)
    h = dt / sub
    out = np.empty((n, len(x)))

    def step(x):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        return x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    for _ in range(burn):
        x = step(x)
    for i in range(n):
        for _ in range(sub):
            x = step(x)
        out[i] = x
    return out


def lorenz(x, s=10.0, r=28.0, b=8.0 / 3.0):
    return np.array([s * (x[1] - x[0]), x[0] * (r - x[2]) - x[1], x[0] * x[1] - b * x[2]])


def rossler(x, a=0.2, b=0.2, c=5.7):
    return np.array([-x[1] - x[2], x[0] + a * x[1], b + x[2] * (x[0] - c)])


def chen(x, a=35.0, b=3.0, c=28.0):
    return np.array([a * (x[1] - x[0]), (c - a) * x[0] - x[0] * x[2] + c * x[1], x[0] * x[1] - b * x[2]])


def vdp(x, mu=1.0):
    return np.array([x[1], mu * (1 - x[0] ** 2) * x[1] - x[0]])


def ar1(n, phi, rng):
    e = rng.standard_normal(n + 500)
    x = np.zeros(n + 500)
    for i in range(1, n + 500):
        x[i] = phi * x[i - 1] + e[i]
    return x[500:]


def ar2_narrowband(n, rng, period=40.0, r=0.98):
    """Linear stochastic oscillator: noise-driven damped resonance."""
    a1 = 2 * r * np.cos(2 * np.pi / period)
    a2 = -r * r
    e = rng.standard_normal(n + 1000)
    x = np.zeros(n + 1000)
    for i in range(2, n + 1000):
        x[i] = a1 * x[i - 1] + a2 * x[i - 2] + e[i]
    return x[1000:]


def noisy_vdp(n, rng, dt=0.05, sigma=0.3):
    """Van der Pol limit cycle with DYNAMICAL noise (Euler-Maruyama)."""
    sub = 10
    h = dt / sub
    x = np.array([1.0, 0.0])
    out = np.empty(n)
    for i in range(-2000, n):
        for _ in range(sub):
            x = x + h * vdp(x) + np.array([0.0, sigma * np.sqrt(h) * rng.standard_normal()])
        if i >= 0:
            out[i] = x[0]
    return out

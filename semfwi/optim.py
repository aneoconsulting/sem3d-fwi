"""L-BFGS two-loop recursion with a diagonal preconditioner; the inversion loop (line search,
stages, regularization) is semfwi.inversion.invert.
"""
import numpy as np


class LBFGS:
    def __init__(self, m=5, precond=None):
        self.m = m
        self.precond = precond or (lambda v: v)
        self.s, self.y = [], []

    def direction(self, g):
        q = g.copy()
        alphas = []
        for s, y in reversed(list(zip(self.s, self.y))):
            a = s @ q / (y @ s)
            alphas.append(a)
            q -= a * y
        r = self.precond(q)
        if self.s:
            s, y = self.s[-1], self.y[-1]
            r *= (s @ y) / (y @ self.precond(y))
        for (s, y), a in zip(zip(self.s, self.y), reversed(alphas)):
            b = y @ r / (y @ s)
            r += (a - b) * s
        return -r

    def update(self, s, y):
        if s @ y <= 1e-12 * np.linalg.norm(s) * np.linalg.norm(y):
            return False
        self.s.append(s)
        self.y.append(y)
        if len(self.s) > self.m:
            self.s.pop(0)
            self.y.pop(0)
        return True

    def reset(self):
        self.s, self.y = [], []

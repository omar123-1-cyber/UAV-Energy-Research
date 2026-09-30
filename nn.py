"""Minimal batched MLP with exact backprop (incl. gradient w.r.t. input) and Adam."""
import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))   # Glorot-uniform
            W = rng.uniform(-lim, lim, (sizes[i], sizes[i + 1]))
            if i == len(sizes) - 2:
                W *= out_scale
            self.W.append(W); self.b.append(np.zeros(sizes[i + 1]))

    @property
    def params(self):
        return self.W + self.b

    def forward(self, x, cache=False):
        acts = [x]; h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        if cache:
            self._acts = acts
        return h

    def backward(self, dout, need_input_grad=False):
        """dout: dLoss/dOutput (batch, out). Returns grads (same order as params) and dL/dx."""
        acts = self._acts
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        g = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(0)
            if i > 0 or need_input_grad:
                g = g @ self.W[i].T
                if i > 0:
                    g = g * (acts[i] > 0)
        return gW + gb, (g if need_input_grad else None)

    def copy_from(self, o):
        self.W = [w.copy() for w in o.W]; self.b = [b.copy() for b in o.b]

    def polyak(self, o, tau):
        for a, b in zip(self.params, o.params):
            a *= (1 - tau); a += tau * b

    def state(self):
        return [p.copy() for p in self.params]

    def load(self, st):
        n = len(self.W)
        self.W = [s.copy() for s in st[:n]]; self.b = [s.copy() for s in st[n:]]


class Adam:
    def __init__(self, params, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, max_norm=None):
        self.lr, self.b1, self.b2, self.eps, self.max_norm = lr, b1, b2, eps, max_norm
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]; self.t = 0

    def step(self, params, grads):
        if self.max_norm is not None:
            n = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
            if n > self.max_norm:
                grads = [g * (self.max_norm / (n + 1e-12)) for g in grads]
        self.t += 1
        c1 = 1 - self.b1 ** self.t; c2 = 1 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)

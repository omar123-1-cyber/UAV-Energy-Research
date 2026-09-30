"""
agents.py -- NumPy implementations of SAC (Haarnoja et al. 2018, automatic temperature),
PPO-clip (Schulman et al. 2017) and DQN (Mnih et al. 2015).
Hyper-parameters follow the Stable-Baselines3 defaults unless stated in HP.
Gradients are exact and verified by finite differences in test_unit.py.
"""
import numpy as np
from nn import MLP, Adam

LOG2PI = np.log(2 * np.pi)
LS_MIN, LS_MAX = -5.0, 2.0


def softplus(x):
    return np.logaddexp(0.0, x)


class Replay:
    def __init__(self, obs_dim, act_dim, cap, rng, act_dtype=np.float64):
        self.s = np.zeros((cap, obs_dim)); self.a = np.zeros((cap, act_dim), act_dtype)
        self.r = np.zeros(cap); self.s2 = np.zeros((cap, obs_dim)); self.d = np.zeros(cap)
        self.cap, self.n, self.i, self.rng = cap, 0, 0, rng

    def add(self, s, a, r, s2, d):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.i = (i + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, bs):
        j = self.rng.integers(0, self.n, bs)
        return self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j]


# =============================================================================== SAC
class SAC:
    HP = dict(hidden=128, lr=3e-4, gamma=0.99, tau=0.005, batch=128, buffer=300_000, learning_starts=5_000, alpha_init=0.1)

    def __init__(self, obs_dim, act_dim, seed=0, **kw):
        hp = dict(self.HP); hp.update(kw); self.hp = hp
        self.rng = np.random.default_rng(seed); h = hp["hidden"]
        self.ad = act_dim
        self.actor = MLP([obs_dim, h, h, 2 * act_dim], self.rng, out_scale=0.1)
        self.q = [MLP([obs_dim + act_dim, h, h, 1], self.rng) for _ in range(2)]
        self.qt = [MLP([obs_dim + act_dim, h, h, 1], self.rng) for _ in range(2)]
        for t, q in zip(self.qt, self.q):
            t.copy_from(q)
        self.opt_a = Adam(self.actor.params, hp["lr"])
        self.opt_q = [Adam(q.params, hp["lr"]) for q in self.q]
        self.log_alpha = float(np.log(hp['alpha_init'])); self.opt_alpha_m = 0.0; self.opt_alpha_v = 0.0; self.alpha_t = 0
        self.target_entropy = -float(act_dim)
        self.buf = Replay(obs_dim, act_dim, hp["buffer"], self.rng)
        self.steps = 0

    # -- policy head
    def _head(self, out):
        mu = out[:, :self.ad]; raw = out[:, self.ad:]
        tr = np.tanh(raw)
        ls = LS_MIN + 0.5 * (LS_MAX - LS_MIN) * (tr + 1)
        return mu, ls, tr

    def sample(self, s, eps):
        out = self.actor.forward(s, cache=True)
        mu, ls, tr = self._head(out)
        std = np.exp(ls); u = mu + std * eps; a = np.tanh(u)
        # log pi(a|s) with tanh correction; log(1 - tanh(u)^2) = 2(log2 - u - softplus(-2u))
        logp = np.sum(-0.5 * eps ** 2 - ls - 0.5 * LOG2PI, 1) - np.sum(2 * (np.log(2) - u - softplus(-2 * u)), 1)
        return a, logp, (mu, ls, tr, std, u, eps)

    def act(self, s, deterministic=False):
        s = s[None]
        if deterministic:
            mu = self.actor.forward(s)[:, :self.ad]
            return np.tanh(mu)[0]
        return self.sample(s, self.rng.standard_normal((1, self.ad)))[0][0]

    def alpha(self):
        return float(np.exp(self.log_alpha))

    def critic_loss_grads(self, S, A, R, S2, D, eps2):
        a2, logp2, _ = self.sample(S2, eps2)
        sa2 = np.concatenate([S2, a2], 1)
        qt = np.minimum(self.qt[0].forward(sa2)[:, 0], self.qt[1].forward(sa2)[:, 0])
        y = R + self.hp["gamma"] * (1 - D) * (qt - self.alpha() * logp2)
        sa = np.concatenate([S, A], 1)
        losses, grads = [], []
        for q in self.q:
            qv = q.forward(sa, cache=True)[:, 0]
            diff = qv - y
            losses.append(0.5 * np.mean(diff ** 2))
            g, _ = q.backward((diff / len(y))[:, None])
            grads.append(g)
        return losses, grads

    def actor_loss_grads(self, S, eps):
        n = len(S); alpha = self.alpha()
        a, logp, (mu, ls, tr, std, u, e) = self.sample(S, eps)
        sa = np.concatenate([S, a], 1)
        qv, dq_da = [], []
        for q in self.q:
            v = q.forward(sa, cache=True)[:, 0]
            _, gin = q.backward(np.ones((n, 1)), need_input_grad=True)
            qv.append(v); dq_da.append(gin[:, -self.ad:])
        use0 = (qv[0] <= qv[1])[:, None]
        qmin = np.where(use0[:, 0], qv[0], qv[1])
        dQ = np.where(use0, dq_da[0], dq_da[1])
        loss = np.mean(alpha * logp - qmin)
        dL_da = -dQ / n
        dL_dlogp = alpha / n
        one_m_a2 = 1 - a ** 2
        # d logp / d mu = 2a ;  d logp / d ls = -1 + 2a * std * eps
        dmu = dL_dlogp * 2 * a + dL_da * one_m_a2
        dls = dL_dlogp * (-1 + 2 * a * std * e) + dL_da * one_m_a2 * std * e
        draw = dls * 0.5 * (LS_MAX - LS_MIN) * (1 - tr ** 2)
        self.actor.forward(S, cache=True)
        g, _ = self.actor.backward(np.concatenate([dmu, draw], 1))
        return loss, g, logp

    def update(self):
        hp = self.hp
        S, A, R, S2, D = self.buf.sample(hp["batch"])
        _, gq = self.critic_loss_grads(S, A, R, S2, D, self.rng.standard_normal((len(S), self.ad)))
        for q, o, g in zip(self.q, self.opt_q, gq):
            o.step(q.params, g)
        _, ga, logp = self.actor_loss_grads(S, self.rng.standard_normal((len(S), self.ad)))
        self.opt_a.step(self.actor.params, ga)
        # temperature (Adam on log alpha)
        g = -np.mean(logp + self.target_entropy)
        self.alpha_t += 1
        self.opt_alpha_m = 0.9 * self.opt_alpha_m + 0.1 * g
        self.opt_alpha_v = 0.999 * self.opt_alpha_v + 0.001 * g * g
        mh = self.opt_alpha_m / (1 - 0.9 ** self.alpha_t); vh = self.opt_alpha_v / (1 - 0.999 ** self.alpha_t)
        self.log_alpha -= hp["lr"] * mh / (np.sqrt(vh) + 1e-8)
        for t, q in zip(self.qt, self.q):
            t.polyak(q, hp["tau"])

    def observe(self, s, a, r, s2, term):
        self.buf.add(s, a, r, s2, float(term)); self.steps += 1
        if self.steps >= self.hp["learning_starts"]:
            self.update()

    def explore_action(self, s):
        if self.steps < self.hp["learning_starts"]:
            return self.rng.uniform(-1, 1, self.ad)
        return self.act(s)

    def policy_state(self):
        return dict(actor=self.actor.state())


# =============================================================================== PPO
class PPO:
    HP = dict(hidden=64, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, n_steps=2048, epochs=10, batch=64,
              ent_coef=0.0, vf_coef=0.5, max_norm=0.5, log_std_init=0.0)

    def __init__(self, obs_dim, act_dim, seed=0, **kw):
        hp = dict(self.HP); hp.update(kw); self.hp = hp
        self.rng = np.random.default_rng(seed); h = hp["hidden"]; self.ad = act_dim
        self.pi = MLP([obs_dim, h, h, act_dim], self.rng, out_scale=0.01)
        self.log_std = np.full(act_dim, hp["log_std_init"])
        self.vf = MLP([obs_dim, h, h, 1], self.rng)
        self.opt_pi = Adam(self.pi.params + [self.log_std], hp["lr"], max_norm=hp["max_norm"])
        self.opt_vf = Adam(self.vf.params, hp["lr"], max_norm=hp["max_norm"])
        self.traj = []; self.steps = 0

    def act(self, s, deterministic=False):
        mu = self.pi.forward(s[None])[0]
        if deterministic:
            return np.clip(mu, -1, 1)
        return mu + np.exp(self.log_std) * self.rng.standard_normal(self.ad)

    def logp(self, mu, a):
        std = np.exp(self.log_std)
        return np.sum(-0.5 * ((a - mu) / std) ** 2 - self.log_std - 0.5 * LOG2PI, 1)

    def policy_loss_grads(self, S, A, ADV, LP_OLD):
        n = len(S); std = np.exp(self.log_std)
        mu = self.pi.forward(S, cache=True)
        lp = self.logp(mu, A)
        ratio = np.exp(lp - LP_OLD)
        clipped = np.clip(ratio, 1 - self.hp["clip"], 1 + self.hp["clip"])
        obj = np.minimum(ratio * ADV, clipped * ADV)
        ent = np.sum(self.log_std + 0.5 + 0.5 * LOG2PI)
        loss = -np.mean(obj) - self.hp["ent_coef"] * ent
        active = (ratio * ADV) <= (clipped * ADV)            # unclipped branch selected by min
        dL_dlp = np.where(active, -ratio * ADV, 0.0) / n
        z = (A - mu) / std
        dmu = dL_dlp[:, None] * z / std
        dls = np.sum(dL_dlp[:, None] * (z ** 2 - 1), 0) - self.hp["ent_coef"]
        g, _ = self.pi.backward(dmu)
        return loss, g + [dls]

    def value_loss_grads(self, S, RET):
        v = self.vf.forward(S, cache=True)[:, 0]
        diff = v - RET
        loss = self.hp["vf_coef"] * np.mean(diff ** 2)
        g, _ = self.vf.backward((2 * self.hp["vf_coef"] * diff / len(S))[:, None])
        return loss, g

    def value(self, s):
        return float(self.vf.forward(s[None])[0, 0])

    def rollout_step(self, s, a_raw, r, s2, term, trunc):
        """store transition; returns True when an update was performed."""
        mu = self.pi.forward(s[None])
        lp = self.logp(mu, a_raw[None])[0]
        v = self.value(s)
        # bootstrap truncated episodes with V(s')
        if trunc and not term:
            r = r + self.hp["gamma"] * self.value(s2)
        self.traj.append((s, a_raw, r, v, lp, float(term or trunc)))
        self.last_s2 = s2; self.last_done = term or trunc
        self.steps += 1
        if len(self.traj) >= self.hp["n_steps"]:
            self.update(); return True
        return False

    def update(self):
        hp = self.hp
        S, A, R, V, LP, Dn = map(np.array, zip(*self.traj))
        nv = 0.0 if self.last_done else self.value(self.last_s2)
        T = len(R); adv = np.zeros(T); gae = 0.0
        for t in reversed(range(T)):
            nxt = nv if t == T - 1 else V[t + 1]
            delta = R[t] + hp["gamma"] * nxt * (1 - Dn[t]) - V[t]
            gae = delta + hp["gamma"] * hp["lam"] * (1 - Dn[t]) * gae
            adv[t] = gae
        ret = adv + V
        for _ in range(hp["epochs"]):
            idx = self.rng.permutation(T)
            for k in range(0, T, hp["batch"]):
                j = idx[k:k + hp["batch"]]
                a = adv[j]; a = (a - a.mean()) / (a.std() + 1e-8)
                _, gp = self.policy_loss_grads(S[j], A[j], a, LP[j])
                self.opt_pi.step(self.pi.params + [self.log_std], gp)
                _, gv = self.value_loss_grads(S[j], ret[j])
                self.opt_vf.step(self.vf.params, gv)
        self.traj = []

    def policy_state(self):
        return dict(pi=self.pi.state(), log_std=self.log_std.copy())


# =============================================================================== DQN
class DQN:
    HP = dict(hidden=128, lr=3e-4, gamma=0.99, batch=64, buffer=100_000, learning_starts=5_000,
              train_freq=4, target_update=1_000, expl_fraction=0.2, eps_final=0.05, max_norm=10.0)

    def __init__(self, obs_dim, n_actions, seed=0, total_steps=150_000, **kw):
        hp = dict(self.HP); hp.update(kw); self.hp = hp
        self.rng = np.random.default_rng(seed); h = hp["hidden"]; self.n = n_actions
        self.q = MLP([obs_dim, h, h, n_actions], self.rng)
        self.qt = MLP([obs_dim, h, h, n_actions], self.rng); self.qt.copy_from(self.q)
        self.opt = Adam(self.q.params, hp["lr"], max_norm=hp["max_norm"])
        self.buf = Replay(obs_dim, 1, hp["buffer"], self.rng, act_dtype=np.int64)
        self.steps = 0; self.total = total_steps

    def eps(self):
        f = min(1.0, self.steps / (self.hp["expl_fraction"] * self.total))
        return 1.0 + f * (self.hp["eps_final"] - 1.0)

    def act(self, s, deterministic=False):
        if not deterministic and self.rng.random() < self.eps():
            return int(self.rng.integers(self.n))
        return int(np.argmax(self.q.forward(s[None])[0]))

    def loss_grads(self, S, A, R, S2, D):
        y = R + self.hp["gamma"] * (1 - D) * self.qt.forward(S2).max(1)
        q = self.q.forward(S, cache=True)
        qa = q[np.arange(len(S)), A]
        diff = qa - y
        huber = np.where(np.abs(diff) <= 1, 0.5 * diff ** 2, np.abs(diff) - 0.5)
        dq = np.zeros_like(q)
        dq[np.arange(len(S)), A] = np.clip(diff, -1, 1) / len(S)
        g, _ = self.q.backward(dq)
        return float(np.mean(huber)), g

    def observe(self, s, a, r, s2, term):
        self.buf.add(s, [a], r, s2, float(term)); self.steps += 1
        hp = self.hp
        if self.steps >= hp["learning_starts"] and self.steps % hp["train_freq"] == 0:
            S, A, R, S2, D = self.buf.sample(hp["batch"])
            _, g = self.loss_grads(S, A[:, 0], R, S2, D)
            self.opt.step(self.q.params, g)
        if self.steps % hp["target_update"] == 0:
            self.qt.copy_from(self.q)

    def policy_state(self):
        return dict(q=self.q.state())

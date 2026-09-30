"""Unit tests: FEA against closed-form solutions, environment physics, and
finite-difference checks of every analytic gradient used by SAC, PPO and DQN.
Run:  python3 test_unit.py"""
import numpy as np, copy, sys
import fea, env
from nn import MLP
from agents import SAC, PPO, DQN

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def fd_check(loss_fn, params, grads, rng, n=12, h=1e-6):
    worst = 0.0
    for _ in range(n):
        k = rng.integers(len(params)); P = params[k]
        idx = tuple(rng.integers(s) for s in P.shape)
        old = P[idx]
        P[idx] = old + h; lp = loss_fn()
        P[idx] = old - h; lm = loss_fn()
        P[idx] = old
        num = (lp - lm) / (2 * h); ana = grads[k][idx]
        err = abs(num - ana) / max(1e-7, abs(num) + abs(ana))
        worst = max(worst, err)
    return worst


rng = np.random.default_rng(0)

# ------------------------------------------------------------------ FEA
p0 = dict(fea.PARAMS); p0["m_tip"] = 0.0
f_fe = fea.modes(p0, 2)[0] / 2 / np.pi
f_an = fea.analytic_f1_no_tip(p0)
check("FEA f1 (no tip mass) vs closed form < 0.01 %", abs(f_fe[0] - f_an) / f_an < 1e-4, f"{f_fe[0]:.4f} vs {f_an:.4f} Hz")
d = fea.derived(p0)
f2_an = 4.694091 ** 2 / 2 / np.pi * np.sqrt(d["EI"] / (d["mA"] * p0["L"] ** 4))
check("FEA f2 (no tip mass) vs closed form < 0.01 %", abs(f_fe[1] - f2_an) / f2_an < 1e-4, f"{f_fe[1]:.3f} vs {f2_an:.3f} Hz")
K, M, le, n = fea.assemble(fea.PARAMS)
Fv = np.zeros(K.shape[0]); Fv[-2] = 1.0
ws = np.linalg.solve(K, Fv)[-2]
check("FEA static tip deflection = FL^3/3EI", abs(ws / fea.analytic_tip_static() - 1) < 1e-6, f"{ws*1e3:.5f} mm")
f_tip = fea.modes(fea.PARAMS, 1)[0][0] / 2 / np.pi
k_tip = 3 * fea.derived()["EI"] / fea.PARAMS["L"] ** 3
m_eff = fea.PARAMS["m_tip"] + 33 / 140 * fea.derived()["mA"] * fea.PARAMS["L"]
f_ray = np.sqrt(k_tip / m_eff) / 2 / np.pi
check("FEA f1 with tip mass vs Rayleigh estimate < 1 %", abs(f_tip / f_ray - 1) < 0.01, f"{f_tip:.3f} vs {f_ray:.3f} Hz")
q = dict(fea.PARAMS); q["delta_1P"] = 0.0
rpm = 6000.0; om = 2 * np.pi * rpm / 60 * 2
Rg = np.logspace(3, 6, 3000)
Pg = [fea.harvest(rpm, 0.026, R, q)["P"] for R in Rg]
R_num = Rg[int(np.argmax(Pg))]; R_th = 1 / (om * fea.derived(q)["Cp"])
check("Electrical: numerical optimum load = 1/(omega Cp)", abs(R_num / R_th - 1) < 0.01, f"{R_num/1e3:.2f} vs {R_th/1e3:.2f} kOhm")
h = fea.harvest(rpm, 0.026, 1e12, q)
check("Electrical: V(R->inf) = V_oc", abs(h["V"][1] / h["Voc"][1] - 1) < 1e-6)
P_hi = fea.harvest(rpm, 0.026, R_th, q)["P"]
Qamp = h["Voc"][1] * fea.derived(q)["Cp"]
check("Electrical: P(R_opt) = omega Q^2 / (4 Cp)", abs(P_hi / (om * Qamp ** 2 / (4 * fea.derived(q)["Cp"])) - 1) < 1e-6)
check("Rotor model: hover rpm reproduces hover thrust",
      abs(fea.derived()["kT"] * (fea.derived()["rpm_hover"] / 60) ** 2 - fea.derived()["T_hover"]) < 1e-9)

# ------------------------------------------------------------------ environment
check("Env: zero command -> hover rpm", abs(env.rpm_of_accel(np.zeros(3)) - env.RPM_HOVER) < 1e-6)
e = env.DroneEnv(seed=1); e.reset(scenario_seed=5)
e.C = np.array([[5.0, 5.0, 2.0]]); e.R = np.array([1.0]); e.p = np.array([2.0, 5.0, 2.0])
r = e.rays(e.p)
check("Env: ray-sphere range (+x) = 2.0 m", abs(r[0] - 2.0) < 1e-9, f"{r[0]:.6f}")
check("Env: ray to ceiling = min(3, 5.5-2)", abs(r[8] - 3.0) < 1e-9)
check("Env: ray to floor = 2 - 0.3", abs(r[9] - 1.7) < 1e-9)
e.p = np.array([5.0 - 1.2, 5.0, 2.0]); e.v = np.zeros(3)
_, _, term, _, info = e.step(np.zeros(3))
check("Env: collision detected inside R + body radius", term and info["crash"])
e = env.DroneEnv(seed=1); e.reset(scenario_seed=5); e.p = e.goal + np.array([0.3, 0, 0]); e.v = np.zeros(3)
_, _, term, _, info = e.step(np.zeros(3))
check("Env: success inside goal tolerance", term and info["success"])
e = env.DroneEnv(seed=1); e.reset(scenario_seed=7); e.C = np.array([[50, 50, 50.0]]); e.R = np.array([0.1])
E0 = e.E
e.step(np.zeros(3))
check("Env: hover step energy = P_hover * dt", abs(e.E - E0 - env.P_HOVER * env.DT) < 1e-9)
check("Env: observation dimension = 17", e.obs().shape == (17,))

# ------------------------------------------------------------------ MLP gradients
net = MLP([5, 7, 6, 3], rng)
X = rng.normal(size=(4, 5)); Wt = rng.normal(size=(4, 3))
lf = lambda: float(np.sum(net.forward(X) * Wt))
net.forward(X, cache=True); g, gx = net.backward(Wt, need_input_grad=True)
check("MLP parameter gradients (finite difference)", fd_check(lf, net.params, g, rng, 30) < 1e-5)
Xc = X.copy(); num = np.zeros_like(X)
for i in range(4):
    for j in range(5):
        X[i, j] += 1e-6; a1 = lf(); X[i, j] -= 2e-6; a2 = lf(); X[i, j] += 1e-6
        num[i, j] = (a1 - a2) / 2e-6
check("MLP input gradients (finite difference)", np.max(np.abs(num - gx)) < 1e-5)

# ------------------------------------------------------------------ SAC
sac = SAC(17, 3, seed=3, hidden=16)
sac.log_alpha = np.log(0.3)
S = rng.normal(size=(8, 17)); A = rng.uniform(-1, 1, (8, 3)); R = rng.normal(size=8)
S2 = rng.normal(size=(8, 17)); Dd = (rng.random(8) < 0.3).astype(float)
eps2 = rng.normal(size=(8, 3)); eps = rng.normal(size=(8, 3))
losses, gq = sac.critic_loss_grads(S, A, R, S2, Dd, eps2)
for i in range(2):
    lf = lambda i=i: sac.critic_loss_grads(S, A, R, S2, Dd, eps2)[0][i]
    check(f"SAC critic {i+1} gradients (finite difference)", fd_check(lf, sac.q[i].params, gq[i], rng, 30) < 1e-5)
loss, ga, _ = sac.actor_loss_grads(S, eps)
lf = lambda: sac.actor_loss_grads(S, eps)[0]
w = fd_check(lf, sac.actor.params, ga, rng, 40)
check("SAC actor reparameterised gradients incl. tanh correction (finite difference)", w < 1e-4, f"worst rel err {w:.2e}")
# log-prob vs change-of-variables density
a, logp, (mu, ls, tr, std, u, ee) = sac.sample(S[:1], eps[:1])
ref = np.sum(-0.5 * ((u - mu) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)) - np.sum(np.log(1 - np.tanh(u) ** 2))
check("SAC log-prob = log N(u) - sum log(1 - tanh^2 u)", abs(logp[0] - ref) < 1e-8)

# ------------------------------------------------------------------ PPO
ppo = PPO(17, 3, seed=4, hidden=16); ppo.log_std[:] = [-0.3, 0.1, 0.2]
S = rng.normal(size=(16, 17)); A = rng.normal(size=(16, 3)); ADV = rng.normal(size=16)
mu = ppo.pi.forward(S); LPo = ppo.logp(mu, A) + rng.normal(0, 0.3, 16)
ppo.hp["ent_coef"] = 0.01
loss, gp = ppo.policy_loss_grads(S, A, ADV, LPo)
lf = lambda: ppo.policy_loss_grads(S, A, ADV, LPo)[0]
w = fd_check(lf, ppo.pi.params + [ppo.log_std], gp, rng, 40)
check("PPO clipped-surrogate gradients incl. log-std (finite difference)", w < 1e-4, f"worst rel err {w:.2e}")
RET = rng.normal(size=16)
loss, gv = ppo.value_loss_grads(S, RET)
lf = lambda: ppo.value_loss_grads(S, RET)[0]
check("PPO value gradients (finite difference)", fd_check(lf, ppo.vf.params, gv, rng, 30) < 1e-5)

# ------------------------------------------------------------------ DQN
dqn = DQN(17, 27, seed=5, hidden=16)
S = rng.normal(size=(8, 17)); A = rng.integers(0, 27, 8); R = rng.normal(size=8) * 3
S2 = rng.normal(size=(8, 17)); Dd = np.zeros(8)
loss, g = dqn.loss_grads(S, A, R, S2, Dd)
lf = lambda: dqn.loss_grads(S, A, R, S2, Dd)[0]
check("DQN Huber TD gradients (finite difference)", fd_check(lf, dqn.q.params, g, rng, 30) < 1e-5)

if __name__ == "__main__":
    npass = sum(ok for _, ok, _ in RESULTS)
    for nme, ok, det in RESULTS:
        print(("PASS " if ok else "FAIL ") + nme + (f"   [{det}]" if det else ""))
    print(f"\n{npass}/{len(RESULTS)} passed")
    import json; json.dump([dict(name=a, ok=b, detail=c) for a, b, c in RESULTS], open("results/tests.json", "w"), indent=1)
    sys.exit(0 if npass == len(RESULTS) else 1)

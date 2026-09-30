"""
env.py -- mapless 3-D obstacle-avoidance task for a point-mass quadrotor with a
physics-based propulsion-energy model and the FEA-derived piezoelectric harvest.

Energy accounting (per 50 ms step):
  thrust vector   T = m (a_cmd + g e_z)           (drag is part of the net acceleration model)
  per-rotor thrust T/4 -> rpm = 60 sqrt(T/4 / k_T)
  propulsion      P = 4 * CP rho n^3 D^5 / eta  + P_avionics
  harvest         P_h = 4 * P_patch(rpm)          (root patch, FEA lookup, design load)
"""
import numpy as np
import fea

P = fea.PARAMS
D = fea.derived(P)
_lk = np.load(__file__.replace("env.py", "results/harvest_lookup.npz"))
_RPM_LK, _PH_LK = _lk["rpm"], _lk["P"]

DT = 0.05
A_MAX = 2.0           # m/s^2 per axis
V_MAX = 4.0
DRAG = 0.3            # 1/s linear drag on net motion
BOX = np.array([10.0, 10.0, 5.5])
Z_MIN = 0.3
N_OBS = 8
R_BODY = 0.25
GOAL_TOL = 0.5
T_MAX_STEPS = 400
RAY_RANGE = 3.0
_dirs = [np.array([np.cos(t), np.sin(t), 0.0]) for t in np.arange(8) * np.pi / 4]
RAY_DIRS = np.array(_dirs + [np.array([0, 0, 1.0]), np.array([0, 0, -1.0])])
OBS_DIM = 3 + 1 + 3 + len(RAY_DIRS)       # 17: goal unit vector, distance, velocity, 10 rays
ACT_DIM = 3

MASS = P["mass"]; G = P["g"]


def rpm_of_accel(a):
    T = MASS * np.linalg.norm(a + np.array([0, 0, G])) / 4.0
    return float(np.clip(fea.rpm_from_thrust(T), 0, P["rpm_max"]))


def prop_power(rpm):
    return 4 * fea.rotor_power(rpm) + P["P_avionics"]


def harvest_power(rpm):
    return 4 * float(np.interp(rpm, _RPM_LK, _PH_LK))


RPM_HOVER = D["rpm_hover"]
P_HOVER = prop_power(RPM_HOVER)
PH_HOVER = harvest_power(RPM_HOVER)

REWARD_MODES = {
    # step, progress, energy, harvest, goal, crash
    "standard":     dict(step=0.01, prog=5.0, energy=0.0,  harvest=0.0,  goal=10.0, crash=-10.0),
    "energy":       dict(step=0.01, prog=5.0, energy=0.10, harvest=0.0,  goal=10.0, crash=-10.0),
    "energy_harv":  dict(step=0.01, prog=5.0, energy=0.10, harvest=0.10, goal=10.0, crash=-10.0),
}


def make_scenario(seed):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        start = np.array([rng.uniform(0.7, 1.5), rng.uniform(1, 9), rng.uniform(1.0, 4.5)])
        goal = np.array([rng.uniform(8.5, 9.3), rng.uniform(1, 9), rng.uniform(1.0, 4.5)])
        C = np.column_stack([rng.uniform(2.5, 7.5, N_OBS), rng.uniform(1, 9, N_OBS), rng.uniform(0.8, 4.5, N_OBS)])
        # bias half the obstacles onto the straight line so the direct path is blocked
        for k in range(N_OBS // 2):
            t = rng.uniform(0.3, 0.7)
            C[k] = start + t * (goal - start) + rng.normal(0, 0.4, 3)
        Rr = rng.uniform(0.5, 0.8, N_OBS)
        ok = all(np.linalg.norm(C - q, axis=1).min() > (Rr.max() + R_BODY + 0.4) for q in (start, goal))
        ok &= np.all(C[:, 2] > 0.5)
        if ok:
            return dict(start=start, goal=goal, C=C, R=Rr)
    raise RuntimeError


class DroneEnv:
    def __init__(self, reward_mode="energy", seed=0, noise_std=0.0, wind_std=0.0, scenario_seeds=None):
        self.w = REWARD_MODES[reward_mode]
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std; self.wind_std = wind_std
        self.scenario_seeds = scenario_seeds

    # --- sensing ---
    def rays(self, p):
        out = np.empty(len(RAY_DIRS))
        for i, u in enumerate(RAY_DIRS):
            t = RAY_RANGE
            # walls / floor / ceiling
            for ax in range(3):
                if u[ax] > 1e-9:
                    t = min(t, (BOX[ax] - p[ax]) / u[ax])
                elif u[ax] < -1e-9:
                    lo = Z_MIN if ax == 2 else 0.0
                    t = min(t, (lo - p[ax]) / u[ax])
            oc = p - self.C
            b = oc @ u
            c = np.sum(oc * oc, 1) - self.R ** 2
            disc = b * b - c
            m = disc >= 0
            if m.any():
                tt = -b[m] - np.sqrt(disc[m])
                tt = tt[tt > 0]
                if tt.size:
                    t = min(t, tt.min())
            out[i] = max(t, 0.0)
        return out

    def obs(self):
        r = self.rays(self.p) / RAY_RANGE
        if self.noise_std > 0:
            r = np.clip(r + self.rng.normal(0, self.noise_std, r.shape), 0, 1)
        g = self.goal - self.p; dist = np.linalg.norm(g)
        return np.concatenate([g / max(dist, 1e-6), [min(dist, 10.0) / 10.0], self.v / V_MAX, r]).astype(np.float64)

    def reset(self, scenario_seed=None):
        if scenario_seed is None:
            scenario_seed = int(self.rng.integers(0, 2 ** 31 - 1)) if self.scenario_seeds is None \
                else int(self.rng.choice(self.scenario_seeds))
        s = make_scenario(scenario_seed)
        self.p, self.goal, self.C, self.R = s["start"].copy(), s["goal"], s["C"], s["R"]
        self.v = np.zeros(3); self.t = 0; self.E = 0.0; self.Eh = 0.0; self.path = 0.0
        self.d = np.linalg.norm(self.goal - self.p); self.d0 = self.d
        self.wind = np.zeros(3); self.rpm_sum = 0.0
        return self.obs()

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1) * A_MAX
        rpm = rpm_of_accel(a)
        e_step = prop_power(rpm) * DT
        h_step = harvest_power(rpm) * DT
        if self.wind_std > 0:   # Ornstein-Uhlenbeck gust acceleration
            self.wind += -0.5 * self.wind * DT + self.wind_std * np.sqrt(DT) * self.rng.normal(size=3)
        acc = a - DRAG * self.v + self.wind
        self.v = self.v + acc * DT
        sp = np.linalg.norm(self.v)
        if sp > V_MAX:
            self.v *= V_MAX / sp
        p_old = self.p.copy()
        self.p = self.p + self.v * DT
        self.path += np.linalg.norm(self.p - p_old)
        self.t += 1; self.E += e_step; self.Eh += h_step; self.rpm_sum += rpm
        d = np.linalg.norm(self.goal - self.p)
        w = self.w
        r = w["prog"] * (self.d - d) - w["step"] - w["energy"] * e_step / (P_HOVER * DT) \
            + w["harvest"] * h_step / (PH_HOVER * DT)
        self.d = d
        crash = bool(np.any(np.linalg.norm(self.C - self.p, axis=1) < self.R + R_BODY)) \
            or np.any(self.p[:2] < 0) or np.any(self.p[:2] > BOX[:2]) or self.p[2] < Z_MIN or self.p[2] > BOX[2]
        success = d < GOAL_TOL
        term = crash or success
        trunc = (not term) and self.t >= T_MAX_STEPS
        if success:
            r += w["goal"]
        elif crash:
            r += w["crash"]
        info = dict(success=bool(success), crash=bool(crash), timeout=bool(trunc), E=self.E, Eh=self.Eh,
                    t=self.t * DT, path=self.path, d0=self.d0, mean_rpm=self.rpm_sum / self.t)
        return self.obs(), float(r), bool(term), bool(trunc), info


def discrete_actions():
    v = np.array([-1.0, 0.0, 1.0])
    return np.array([[a, b, c] for a in v for b in v for c in v])

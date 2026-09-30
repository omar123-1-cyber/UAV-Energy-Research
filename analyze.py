"""Aggregate all runs: statistics, robustness, baseline, figures -> results/drl.json, figs/*.pdf"""
import json, glob, pickle
import numpy as np
from scipy import stats
import env as E, baseline as B
from train import TEST_SEEDS, evaluate
from agents import SAC, PPO, DQN
from style import plt, C, GRAY, INK2, panel

SEEDS = [0, 1, 2, 3, 4]
DACT = E.discrete_actions()


def load(algo, mode, seed):
    return json.load(open(f"results/runs/{algo}_{mode}_s{seed}.json"))


def metrics(test):
    s = np.array([i["success"] for i in test])
    succ = [i for i in test if i["success"]]
    return dict(
        SR=100 * s.mean(), crash=100 * np.mean([i["crash"] for i in test]), timeout=100 * np.mean([i["timeout"] for i in test]),
        E_succ=np.mean([i["E"] for i in succ]) if succ else np.nan,
        J_per_m=np.mean([i["E"] / i["d0"] for i in succ]) if succ else np.nan,
        t_succ=np.mean([i["t"] for i in succ]) if succ else np.nan,
        Eh_uJ=1e6 * np.mean([i["Eh"] for i in succ]) if succ else np.nan,
        Eh_per_s_uW=1e6 * np.mean([i["Eh"] / i["t"] for i in succ]) if succ else np.nan,
        rpm=np.mean([i["mean_rpm"] for i in succ]) if succ else np.nan,
        speed=np.mean([i["path"] / i["t"] for i in succ]) if succ else np.nan,
    )


def ms(vals):
    v = np.array(vals, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), per_seed=v.tolist(),
                ci95=[float(x) for x in stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))])


def hedges_g(a, b):
    a, b = np.asarray(a), np.asarray(b); na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.zeros(m); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * pvals[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def compare(groups, keys, metric):
    names = list(keys); out = {}
    if len(names) > 2:
        F, p = stats.f_oneway(*[groups[n][metric] for n in names])
        out["anova"] = dict(F=float(F), p=float(p))
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    ps = []; rows = []
    for a, b in pairs:
        t, p = stats.ttest_ind(groups[a][metric], groups[b][metric], equal_var=False)
        ps.append(p); rows.append(dict(a=a, b=b, diff=float(np.mean(groups[a][metric]) - np.mean(groups[b][metric])),
                                       t=float(t), p=float(p), g=hedges_g(groups[a][metric], groups[b][metric])))
    for r, pa in zip(rows, holm(ps)):
        r["p_holm"] = pa
    out["pairs"] = rows
    return out


res = {}
# -------------------------------------------------------------- per-run metrics
runs = {}
for algo, mode in [("SAC", "energy"), ("PPO", "energy"), ("DQN", "energy"), ("SAC", "standard"), ("SAC", "energy_harv")]:
    per = [metrics(load(algo, mode, s)["test"]) for s in SEEDS]
    runs[f"{algo}-{mode}"] = {k: [p[k] for p in per] for k in per[0]}
res["runs"] = {name: {k: ms(v) for k, v in d.items()} for name, d in runs.items()}

# -------------------------------------------------------------- baseline on the identical test set
e = E.DroneEnv("energy", seed=0)
astar = [B.run(e, sd) for sd in TEST_SEEDS]
res["astar"] = metrics(astar)
res["astar"]["no_path"] = int(sum(i.get("no_path", False) for i in astar))

# -------------------------------------------------------------- statistics
res["stats_algos"] = {m: compare(runs, ["SAC-energy", "PPO-energy", "DQN-energy"], m) for m in ("SR", "J_per_m")}
res["stats_reward"] = {m: compare(runs, ["SAC-standard", "SAC-energy", "SAC-energy_harv"], m)
                       for m in ("SR", "J_per_m", "Eh_per_s_uW", "rpm", "speed", "t_succ")}

# -------------------------------------------------------------- physical harvest credit (analytic)
rpm = np.linspace(4000, 6500, 200)
res["harvest_to_propulsion_max_ratio"] = float(max(E.harvest_power(r) / E.prop_power(r) for r in rpm))
res["hover"] = dict(P_prop=E.P_HOVER, P_harv=E.PH_HOVER, rpm=E.RPM_HOVER)


# -------------------------------------------------------------- robustness (no retraining)
def load_policy(algo, mode, seed):
    st = pickle.load(open(f"results/models/{algo}_{mode}_s{seed}.pkl", "rb"))
    if algo == "SAC":
        ag = SAC(E.OBS_DIM, E.ACT_DIM); ag.actor.load(st["actor"]); return lambda s: ag.act(s, True)
    if algo == "PPO":
        ag = PPO(E.OBS_DIM, E.ACT_DIM); ag.pi.load(st["pi"]); return lambda s: ag.act(s, True)
    ag = DQN(E.OBS_DIM, len(DACT)); ag.q.load(st["q"]); return lambda s: DACT[ag.act(s, True)]


conds = [("nominal", 0.0, 0.0), ("ray noise σ=0.05", 0.05, 0.0), ("ray noise σ=0.10", 0.10, 0.0),
         ("wind σ=0.5", 0.0, 0.5), ("wind σ=1.0", 0.0, 1.0), ("noise 0.05 + wind 0.5", 0.05, 0.5)]
ROB_SEEDS = TEST_SEEDS[:100]
rob = {}
for name, nz, wd in conds:
    srs = []
    for s in SEEDS:
        pol = load_policy("SAC", "energy", s)
        srs.append(100 * np.mean([i["success"] for i in evaluate(pol, ROB_SEEDS, "energy", nz, wd, eval_seed=77 + s)]))
    ea = E.DroneEnv("energy", seed=77, noise_std=nz, wind_std=wd)
    a_sr = 100 * np.mean([B.run(ea, sd)["success"] for sd in ROB_SEEDS])
    rob[name] = dict(SAC=ms(srs), astar=a_sr, noise=nz, wind=wd)
    print("robust", name, rob[name]["SAC"]["mean"], a_sr, flush=True)
res["robustness"] = rob

json.dump(res, open("results/drl.json", "w"), indent=1, default=float)

# ============================================================== figures
# learning curves
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axs[0]
for i, (name, lab) in enumerate([("SAC-energy", "SAC"), ("PPO-energy", "PPO"), ("DQN-energy", "DQN")]):
    a, m = name.split("-")
    cur = [load(a, m, s)["curve"] for s in SEEDS]
    x = np.array([c["step"] for c in cur[0]]) / 1e3
    y = 100 * np.array([[c["val_success"] for c in cc] for cc in cur])
    ax.plot(x, y.mean(0), color=C[i], label=lab, ls=["-", "--", "-."][i])
    ax.fill_between(x, y.mean(0) - y.std(0, ddof=1), y.mean(0) + y.std(0, ddof=1), color=C[i], alpha=0.15, lw=0)
ax.set_xlabel("environment steps (×10³)"); ax.set_ylabel("validation success (%)"); ax.set_ylim(0, 100)
ax.legend(loc="lower right"); panel(ax, "a")
ax = axs[1]
for i, (name, lab) in enumerate([("SAC-standard", "standard"), ("SAC-energy", "energy-aware"), ("SAC-energy_harv", "energy-aware + harvest bonus")]):
    a, m = name.split("-")
    cur = [load(a, m, s)["curve"] for s in SEEDS]
    x = np.array([c["step"] for c in cur[0]]) / 1e3
    y = 100 * np.array([[c["val_success"] for c in cc] for cc in cur])
    ax.plot(x, y.mean(0), color=C[i], label=lab, ls=["-", "--", "-."][i])
    ax.fill_between(x, y.mean(0) - y.std(0, ddof=1), y.mean(0) + y.std(0, ddof=1), color=C[i], alpha=0.15, lw=0)
ax.set_xlabel("environment steps (×10³)"); ax.set_ylabel("validation success (%)"); ax.set_ylim(0, 100)
ax.legend(loc="lower right"); panel(ax, "b")
fig.tight_layout(); fig.savefig("figs/fig_curves.pdf"); fig.savefig("figs/fig_curves.png")

# trade-off: SR vs J/m per seed, all methods
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
ax = axs[0]
items = [("SAC-energy", "SAC", C[0], "o"), ("PPO-energy", "PPO", C[1], "s"), ("DQN-energy", "DQN", C[2], "^")]
for name, lab, col, mk in items:
    ax.scatter(runs[name]["J_per_m"], runs[name]["SR"], color=col, marker=mk, s=28, label=lab, edgecolor="white", linewidth=0.6, zorder=3)
ax.scatter([res["astar"]["J_per_m"]], [res["astar"]["SR"]], color=GRAY, marker="D", s=30, label="A*+PD (map)", zorder=3)
ax.set_xlabel("propulsion energy per metre of goal distance (J/m)"); ax.set_ylabel("test success (%)")
ax.legend(loc="lower left", fontsize=7); panel(ax, "a")
ax = axs[1]
items = [("SAC-standard", "standard", C[0], "o"), ("SAC-energy", "energy-aware", C[1], "s"), ("SAC-energy_harv", "+ harvest bonus", C[2], "^")]
for name, lab, col, mk in items:
    ax.scatter(runs[name]["J_per_m"], runs[name]["SR"], color=col, marker=mk, s=28, label=lab, edgecolor="white", linewidth=0.6, zorder=3)
ax.set_xlabel("propulsion energy per metre of goal distance (J/m)"); ax.set_ylabel("test success (%)")
ax.legend(loc="lower left", fontsize=7); panel(ax, "b")
fig.tight_layout(); fig.savefig("figs/fig_tradeoff.pdf"); fig.savefig("figs/fig_tradeoff.png")

# robustness
fig, ax = plt.subplots(figsize=(3.4, 2.5))
names = list(rob)
x = np.arange(len(names))
m = [rob[n]["SAC"]["mean"] for n in names]; sd = [rob[n]["SAC"]["sd"] for n in names]
ax.bar(x - 0.18, m, 0.34, yerr=sd, color=C[0], label="SAC (mapless)", error_kw=dict(lw=0.8, capsize=2))
ax.bar(x + 0.18, [rob[n]["astar"] for n in names], 0.34, color=GRAY, label="A*+PD (map)")
ax.set_xticks(x); ax.set_xticklabels([n.replace(" + ", "\n+ ").replace("ray noise ", "noise\n").replace("wind ", "wind\n") for n in names], fontsize=6.5)
ax.set_ylabel("success (%)"); ax.set_ylim(0, 105); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.8)
fig.tight_layout(); fig.savefig("figs/fig_robust.pdf"); fig.savefig("figs/fig_robust.png")

# example trajectories: first test scenario in which all three SAC variants (seed 0) succeed
def rollout(algo, mode, sd):
    pol = load_policy(algo, mode, 0); ee = E.DroneEnv(mode, seed=0); s = ee.reset(sd); tr = [ee.p.copy()]; done = False
    while not done:
        s, r, te, tru, inf = ee.step(pol(s)); tr.append(ee.p.copy()); done = te or tru
    return np.array(tr), inf
variants = [("SAC standard", "SAC", "standard"), ("SAC energy-aware", "SAC", "energy"), ("SAC + harvest bonus", "SAC", "energy_harv")]
for sd0 in TEST_SEEDS:
    paths = {lab: rollout(a, m, sd0) for lab, a, m in variants}
    if all(inf["success"] for _, inf in paths.values()):
        break
res["traj_example_seed"] = sd0
sc = E.make_scenario(sd0); pa = B.plan(sc)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw=dict(width_ratios=[1, 1]))
for k, (ax, (i, j), yl, ylim) in enumerate(zip(axs, [(0, 1), (0, 2)], ["y (m)", "z (m)"], [(0, 10), (0, 5.5)])):
    for c, r in zip(sc["C"], sc["R"]):
        ax.add_patch(plt.Circle((c[i], c[j]), r, color="#d8d7d2", zorder=1))
    for n, (lab, (tr, inf)) in enumerate(paths.items()):
        ax.plot(tr[:, i], tr[:, j], color=C[n], ls=["-", "--", "-."][n], label=f"{lab} ({inf['E']:.0f} J, {inf['t']:.1f} s)", zorder=3)
    if pa is not None:
        ax.plot(pa[:, i], pa[:, j], color=GRAY, lw=1.0, ls=":", label="A* path (exact map)", zorder=2)
    ax.plot(*sc["start"][[i, j]], "o", color="black", ms=4, zorder=4); ax.plot(*sc["goal"][[i, j]], "*", color="black", ms=9, zorder=4)
    ax.set_xlim(0, 10); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel(yl)
    panel(ax, "ab"[k])
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig("figs/fig_traj.pdf"); fig.savefig("figs/fig_traj.png")
json.dump(res, open("results/drl.json", "w"), indent=1, default=float)
print(json.dumps({k: {kk: vv["mean"] for kk, vv in v.items()} for k, v in res["runs"].items()}, indent=1))
print("astar", res["astar"])
print(json.dumps(res["stats_algos"], indent=1)); print(json.dumps(res["stats_reward"], indent=1))

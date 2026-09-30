"""Supplementary figures, tables and CSV files, generated only from existing results."""
import json, os, csv
import numpy as np
import env as E, baseline as B
from train import TEST_SEEDS
from style import plt, C, GRAY, INK2, panel

os.makedirs("supp", exist_ok=True)
F = json.load(open("results/fea.json")); D = json.load(open("results/drl.json"))
SEEDS = range(5)
CONFIGS = [("SAC", "energy", "SAC, energy-aware"), ("PPO", "energy", "PPO, energy-aware"), ("DQN", "energy", "DQN, energy-aware"),
           ("SAC", "standard", "SAC, standard"), ("SAC", "energy_harv", "SAC, energy-aware + harvest bonus")]
runs = {(a, m, s): json.load(open(f"results/runs/{a}_{m}_s{s}.json")) for a, m, _ in CONFIGS for s in SEEDS}
T = []   # tex macros / tables

# ------------------------------------------------ CSV: per-episode + per-seed
e = E.DroneEnv("energy", seed=0)
astar = [B.run(e, sd) for sd in TEST_SEEDS]
cols = ["method", "algorithm", "reward", "seed", "scenario_seed", "success", "crash", "timeout", "propulsion_energy_J",
        "harvested_energy_uJ", "flight_time_s", "path_length_m", "straight_line_distance_m", "energy_per_m_J", "mean_rotor_rpm"]
with open("supp/ESM_3_per_episode_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols)
    for (a, m, s), r in runs.items():
        for sd, i in zip(TEST_SEEDS, r["test"]):
            w.writerow([f"{a}-{m}", a, m, s, sd, int(i["success"]), int(i["crash"]), int(i["timeout"]), round(i["E"], 3),
                        round(i["Eh"] * 1e6, 5), round(i["t"], 2), round(i["path"], 3), round(i["d0"], 3),
                        round(i["E"] / i["d0"], 3), round(i["mean_rpm"], 1)])
    for sd, i in zip(TEST_SEEDS, astar):
        w.writerow(["A*+PD", "A*+PD", "map-based", "", sd, int(i["success"]), int(i["crash"]), int(i["timeout"]), round(i["E"], 3),
                    round(i["Eh"] * 1e6, 5), round(i["t"], 2), round(float(i["path"]), 3), round(float(i["d0"]), 3),
                    round(i["E"] / i["d0"], 3), round(i["mean_rpm"], 1)])


def summ(test):
    s = [i for i in test if i["success"]]
    return dict(SR=100 * np.mean([i["success"] for i in test]), crash=100 * np.mean([i["crash"] for i in test]),
                timeout=100 * np.mean([i["timeout"] for i in test]), Jm=np.mean([i["E"] / i["d0"] for i in s]),
                t=np.mean([i["t"] for i in s]), v=np.mean([i["path"] / i["t"] for i in s]),
                rpm=np.mean([i["mean_rpm"] for i in s]), Eh=1e6 * np.mean([i["Eh"] / i["t"] for i in s]),
                wall=None)


rows_tex = []
with open("supp/ESM_4_per_seed_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["configuration", "seed", "success_pct", "crash_pct", "timeout_pct", "energy_per_m_J", "flight_time_s",
                "speed_m_s", "mean_rotor_rpm", "harvest_power_uW", "training_wall_time_min"])
    for a, m, lab in CONFIGS:
        for s in SEEDS:
            r = runs[(a, m, s)]; x = summ(r["test"])
            w.writerow([lab, s, round(x["SR"], 1), round(x["crash"], 1), round(x["timeout"], 1), round(x["Jm"], 2), round(x["t"], 2),
                        round(x["v"], 3), round(x["rpm"], 1), round(x["Eh"], 4), round(r["wall_min"], 1)])
            rows_tex.append(f"{lab if s == 0 else ''} & {s} & {x['SR']:.1f} & {x['crash']:.1f} & {x['timeout']:.1f} & {x['Jm']:.1f} & {x['t']:.2f} & {x['v']:.2f} & {x['rpm']:,.0f} & {x['Eh']:.3f} \\\\")
        rows_tex.append("\\midrule")
    x = summ(astar)
    rows_tex.append(f"A*+PD (exact map) & -- & {x['SR']:.1f} & {x['crash']:.1f} & {x['timeout']:.1f} & {x['Jm']:.1f} & {x['t']:.2f} & {x['v']:.2f} & {x['rpm']:,.0f} & {x['Eh']:.3f} \\\\")
open("supp/tab_perseed.tex", "w").write("\n".join(rows_tex) + "\n")

# ------------------------------------------------ Table: FE verification & convergence
v = F["verification"]
rows = [f"$f_1$, no tip mass & {v['f1_no_tip_analytic']:.4f} & {v['f_no_tip_FE'][0]:.4f} & {abs(v['f_no_tip_FE'][0]/v['f1_no_tip_analytic']-1)*100:.1e} \\\\",
        f"$f_2$, no tip mass & {v['f2_no_tip_analytic']:.4f} & {v['f_no_tip_FE'][1]:.4f} & {abs(v['f_no_tip_FE'][1]/v['f2_no_tip_analytic']-1)*100:.1e} \\\\",
        f"Static tip deflection, 1 N (mm) & {v['tip_static_analytic']*1e3:.6f} & {v['tip_static_FE']*1e3:.6f} & {v['static_err_pct']:.1e} \\\\",
        f"$f_1$ with tip mass (Rayleigh) & {v['f1_tip_rayleigh']:.4f} & {F['modes_with_tip'][0]:.4f} & {abs(F['modes_with_tip'][0]/v['f1_tip_rayleigh']-1)*100:.2f} \\\\"]
open("supp/tab_verif.tex", "w").write("\n".join(rows).replace("e-0", "e-") + "\n")
rows = [f"{n} & " + " & ".join(f"{x:.4f}" for x in fr) + " \\\\" for n, fr in v["convergence"].items()]
open("supp/tab_conv.tex", "w").write("\n".join(rows) + "\n")

# robustness per seed
rows = []
for c, r in D["robustness"].items():
    rows.append(c.replace("σ", "$\\sigma$") + " & " + " & ".join(f"{x:.0f}" for x in r["SAC"]["per_seed"]) + f" & {r['SAC']['mean']:.1f} & {r['astar']:.1f} \\\\")
open("supp/tab_rob.tex", "w").write("\n".join(rows) + "\n")

# ------------------------------------------------ Fig S1: voltage & power ratio vs rpm
rs = F["rpm_sweep"]; rpm = np.array(rs["rpm"])
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axs[0]
ax.plot(rpm, rs["Voc1_mV"], color=C[0], label="1P harmonic"); ax.plot(rpm, rs["Voc2_mV"], color=C[1], ls="--", label="2P harmonic")
ax.axhline(2700, color=GRAY, ls=":", lw=0.9); ax.text(3100, 2000, "LTC3588-1 minimum input (2.7 V)", fontsize=6.8, color=INK2)
ax.set_yscale("log"); ax.set_xlabel("motor speed (rpm)"); ax.set_ylabel("open-circuit amplitude (mV)"); ax.legend(loc="center left"); panel(ax, "a")
ax = axs[1]
Ph = 4 * np.array(rs["P_uW"]) * 1e-6; Pp = 4 * np.array(rs["P_rotor_W"]) + F["params"]["P_avionics"]
ax.plot(rpm, Ph / Pp, color=C[0]); ax.set_yscale("log")
ax.set_xlabel("motor speed (rpm)"); ax.set_ylabel("harvest / propulsion power"); panel(ax, "b")
fig.tight_layout(); fig.savefig("supp/figS1.pdf")

# ------------------------------------------------ Fig S2: per-seed learning curves
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.4), sharex=True, sharey=True)
for k, (a, m, lab) in enumerate(CONFIGS):
    ax = axs.flat[k]
    for s in SEEDS:
        cu = runs[(a, m, s)]["curve"]
        ax.plot([c["step"] / 1e3 for c in cu], [100 * c["val_success"] for c in cu], color=C[s], lw=1.1, label=f"seed {s}")
    ax.set_title(lab, fontsize=7.5); ax.set_ylim(0, 100)
    if k % 3 == 0: ax.set_ylabel("validation success (%)")
    if k >= 2: ax.set_xlabel("steps (×10³)")
axs.flat[5].axis("off")
h, l = axs.flat[0].get_legend_handles_labels(); axs.flat[5].legend(h, l, loc="center", fontsize=8)
fig.tight_layout(); fig.savefig("supp/figS2.pdf")

# ------------------------------------------------ Fig S3: training return curves
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
for pi, group in enumerate([CONFIGS[:3], [CONFIGS[3], CONFIGS[0], CONFIGS[4]]]):
    ax = axs[pi]
    for n, (a, m, lab) in enumerate(group):
        cur = [runs[(a, m, s)]["curve"] for s in SEEDS]
        x = np.array([c["step"] for c in cur[0]]) / 1e3
        y = np.array([[c["train_return"] if c["train_return"] is not None else np.nan for c in cc] for cc in cur])
        ax.plot(x, np.nanmean(y, 0), color=C[n], ls=["-", "--", "-."][n], label=lab)
        ax.fill_between(x, np.nanmean(y, 0) - np.nanstd(y, 0), np.nanmean(y, 0) + np.nanstd(y, 0), color=C[n], alpha=0.15, lw=0)
    ax.set_xlabel("steps (×10³)"); ax.set_ylabel("training episode return"); ax.legend(fontsize=6.5, loc="lower right"); panel(ax, "ab"[pi])
fig.tight_layout(); fig.savefig("supp/figS3.pdf")

# ------------------------------------------------ Fig S4: episode energy distributions
fig, ax = plt.subplots(figsize=(7.0, 2.8))
data = []; labs = []
for a, m, lab in CONFIGS:
    data.append([i["E"] / i["d0"] for s in SEEDS for i in runs[(a, m, s)]["test"] if i["success"]]); labs.append(lab.replace(", ", "\n"))
data.append([i["E"] / i["d0"] for i in astar if i["success"]]); labs.append("A*+PD\n(exact map)")
bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color="black", lw=1))
for pbox, col in zip(bp["boxes"], [C[0], C[1], C[2], C[4], C[3], GRAY]):
    pbox.set_facecolor(col); pbox.set_alpha(0.6); pbox.set_edgecolor(INK2)
ax.set_xticks(range(1, len(labs) + 1)); ax.set_xticklabels(labs, fontsize=6.8)
ax.set_ylabel("energy per metre (J/m)")
fig.tight_layout(); fig.savefig("supp/figS4.pdf")

# ------------------------------------------------ Fig S5: six test scenarios with seed-0 SAC trajectories
import pickle
from agents import SAC
st = pickle.load(open("results/models/SAC_energy_s0.pkl", "rb")); ag = SAC(E.OBS_DIM, E.ACT_DIM); ag.actor.load(st["actor"])
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.8))
for k, sd in enumerate(TEST_SEEDS[:6]):
    ax = axs.flat[k]; sc = E.make_scenario(sd); ee = E.DroneEnv("energy", seed=0); s = ee.reset(sd); tr = [ee.p.copy()]; done = False
    while not done:
        s, r, te, tru, inf = ee.step(ag.act(s, True)); tr.append(ee.p.copy()); done = te or tru
    tr = np.array(tr)
    for c, rr in zip(sc["C"], sc["R"]):
        ax.add_patch(plt.Circle((c[0], c[1]), rr, color="#d8d7d2"))
    pa = B.plan(sc)
    ax.plot(pa[:, 0], pa[:, 1], color=GRAY, ls=":", lw=1)
    ax.plot(tr[:, 0], tr[:, 1], color=C[0] if inf["success"] else C[7], lw=1.4)
    ax.plot(*sc["start"][:2], "o", color="black", ms=3); ax.plot(*sc["goal"][:2], "*", color="black", ms=7)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
    ax.set_title(f"scenario {sd}: {'success' if inf['success'] else ('collision' if inf['crash'] else 'time-out')}", fontsize=7.5)
    ax.tick_params(labelsize=6.5)
fig.tight_layout(); fig.savefig("supp/figS5.pdf")
print("done")

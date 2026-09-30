"""Run every FEA analysis reported in the manuscript; write results/fea.json and figures."""
import json, copy
import numpy as np
import fea
from fea import PARAMS, derived, modes, harvest, rotor_power, analytic_f1_no_tip, analytic_tip_static, assemble
from style import plt, C, GRAY, INK2, panel

p = copy.deepcopy(PARAMS); d = derived(p)
res = {"params": p, "derived": {k: float(v) for k, v in d.items()}}

# ---------------- verification ------------------------------------------------
p0 = dict(p); p0["m_tip"] = 0.0
wn0 = modes(p0, 3)[0] / 2 / np.pi
f1_an = analytic_f1_no_tip(p0)
K, M, le, n = assemble(p)
F = np.zeros(K.shape[0]); F[-2] = 1.0
w_static = np.linalg.solve(K, F)[-2]
conv = {}
for ne in (10, 20, 40, 80, 245):
    q = dict(p); q["n_elem"] = ne
    conv[ne] = (modes(q, 3)[0] / 2 / np.pi).tolist()
# tip-mass check: Rayleigh-type estimate with 33/140 of beam mass lumped at tip
k_tip = 3 * d["EI"] / p["L"] ** 3
m_eff = p["m_tip"] + 33 / 140 * d["mA"] * p["L"]
res["verification"] = dict(
    f_no_tip_FE=wn0.tolist(), f1_no_tip_analytic=f1_an,
    f1_err_pct=abs(wn0[0] - f1_an) / f1_an * 100,
    f2_no_tip_analytic=4.694091 ** 2 / 2 / np.pi * np.sqrt(d["EI"] / (d["mA"] * p["L"] ** 4)),
    tip_static_FE=w_static, tip_static_analytic=analytic_tip_static(p),
    static_err_pct=abs(w_static - analytic_tip_static(p)) / analytic_tip_static(p) * 100,
    f1_tip_rayleigh=np.sqrt(k_tip / m_eff) / 2 / np.pi,
    convergence=conv,
)
fn = modes(p, 4)[0] / 2 / np.pi
res["modes_with_tip"] = fn.tolist()

# resonance RPMs
res["rpm_resonance"] = {"1P_f2": fn[1] * 60, "2P_f2": fn[1] * 60 / 2, "2P_f1": fn[0] * 60 / 2}

# ---------------- operating points ---------------------------------------------
rpm_h = d["rpm_hover"]; rpm_max = p["rpm_max"]
ops = {"hover": rpm_h, "cruise": 6000.0, "climb": 7000.0, "max": rpm_max}
res["operating_points"] = ops

# ---------------- optimal load --------------------------------------------------
x_root = p["Lp"] / 2 + 0.001          # patch starts 1 mm from the clamp
R_grid = np.logspace(3, 6.5, 200)
Rsweep = {}
for name, rpm in ops.items():
    P = np.array([harvest(rpm, x_root, R)["P"] for R in R_grid])
    Rsweep[name] = dict(R_opt=float(R_grid[P.argmax()]), P_max=float(P.max()), P=P.tolist())
res["R_sweep"] = {k: {kk: vv for kk, vv in v.items() if kk != "P"} for k, v in Rsweep.items()}
# design resistance: maximise mission-average power over a representative profile
profile = {"hover": 0.55, "cruise": 0.30, "climb": 0.12, "max": 0.03}
res["profile"] = profile
Pmis = np.array([sum(w * harvest(ops[k], x_root, R)["P"] for k, w in profile.items()) for R in R_grid])
R_des = float(R_grid[Pmis.argmax()])
res["R_design"] = R_des
res["R_opt_formula"] = {k: 1 / (2 * np.pi * ops[k] / 60 * 2 * d["Cp"]) for k in ops}  # 2P harmonic

# ---------------- patch location sweep -----------------------------------------
xc_grid = np.linspace(p["Lp"] / 2 + 0.001, p["L"] - p["Lp"] / 2 - 0.001, 120)
loc = {}
for name, rpm in ops.items():
    loc[name] = [harvest(rpm, x, R_des)["P"] for x in xc_grid]
res["location_sweep"] = {"xc_mm": (xc_grid * 1e3).tolist(), **{k: v for k, v in loc.items()}}
named = {"root (P-R)": x_root, "mid-span (P-M)": p["L"] / 2, "motor mount (P-T)": p["L"] - p["Lp"] / 2 - 0.001}
tab = {}
for lab, x in named.items():
    row = {}
    for name, rpm in ops.items():
        h = harvest(rpm, x, R_des)
        row[name] = dict(P_uW=h["P"] * 1e6, Voc_1P_mV=h["Voc"][0] * 1e3, Voc_2P_mV=h["Voc"][1] * 1e3,
                         strain_1P_ue=h["strain"][0] * 1e6, strain_2P_ue=h["strain"][1] * 1e6)
    row["mission_avg_uW"] = sum(w * row[k]["P_uW"] for k, w in profile.items())
    row["x_c_mm"] = x * 1e3
    tab[lab] = row
res["patch_table"] = tab

# ---------------- RPM sweep at root -------------------------------------------
rpm_grid = np.linspace(3000, rpm_max, 300)
rs = [harvest(r, x_root, R_des) for r in rpm_grid]
res["rpm_sweep"] = dict(rpm=rpm_grid.tolist(), P_uW=[r["P"] * 1e6 for r in rs],
                        Voc1_mV=[r["Voc"][0] * 1e3 for r in rs], Voc2_mV=[r["Voc"][1] * 1e3 for r in rs],
                        P_rotor_W=[rotor_power(r) for r in rpm_grid])
# lookup table used by the DRL environment (per patch, root location, R_des)
np.savez("results/harvest_lookup.npz", rpm=rpm_grid, P=np.array([r["P"] for r in rs]))

# ---------------- mission energy budget ---------------------------------------
T_mis = 600.0
P4 = 4 * tab["root (P-R)"]["mission_avg_uW"] * 1e-6
P_prop = 4 * sum(w * rotor_power(ops[k]) for k, w in profile.items()) + p["P_avionics"]
res["mission"] = dict(duration_s=T_mis, P_harvest_4patch_W=P4, E_harvest_J=P4 * T_mis,
                      P_propulsion_W=P_prop, E_propulsion_J=P_prop * T_mis,
                      ratio=P4 / P_prop,
                      LTC3588_quiescent_W_at_5V=950e-9 * 5.0)

# ---------------- sensitivity ---------------------------------------------------
def mission_power(q):
    x = q["Lp"] / 2 + 0.001
    Rg = np.logspace(3, 6.5, 60)
    ops_q = dict(ops);
    return max(sum(w * harvest(ops_q[k], x, R, q)["P"] for k, w in profile.items()) for R in Rg) * 1e6

sens = {"baseline": mission_power(p)}
for key, vals in {"m_tip": [0.0, 0.050, 0.100], "h": [0.004, 0.008, 0.010], "zeta": [0.01, 0.04],
                  "delta_2P": [0.01, 0.04], "delta_1P": [0.005, 0.02]}.items():
    for v in vals:
        q = dict(p); q[key] = v
        f = (modes(q, 2)[0] / 2 / np.pi).tolist()
        sens[f"{key}={v}"] = dict(P_uW=mission_power(q), f1=f[0], f2=f[1])
res["sensitivity"] = sens

json.dump(res, open("results/fea.json", "w"), indent=1, default=float)

# ================= figures =====================================================
# Fig: mode shapes + location sweep + rpm sweep + R sweep
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
ax = axs[0, 0]
wn, phi, le_, n_ = modes(p, 3)
xs = np.arange(n_ + 1) * le_ * 1e3
for i in range(3):
    w = np.concatenate([[0], phi[0::2, i]]); w = w / np.abs(w).max()
    ax.plot(xs, w, color=C[i], ls=["-", "--", ":"][i], label=f"mode {i+1}: {wn[i]/2/np.pi:.1f} Hz")
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_xlabel("position from clamp $x$ (mm)"); ax.set_ylabel("normalised deflection")
ax.legend(loc="lower left"); panel(ax, "a")

ax = axs[0, 1]
for i, name in enumerate(ops):
    ax.plot(res["location_sweep"]["xc_mm"], np.array(loc[name]) * 1e6, color=C[i], ls=["-", "--", "-.", ":"][i],
            label=f"{name} ({ops[name]:.0f} rpm)")
ax.set_yscale("log"); ax.set_xlabel("patch centre $x_c$ (mm)"); ax.set_ylabel("power per patch (µW)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.8); panel(ax, "b")

ax = axs[1, 0]
ax.plot(rpm_grid, res["rpm_sweep"]["P_uW"], color=C[0], label="root patch")
for key, ls in (("2P_f2", "--"),):
    ax.axvline(res["rpm_resonance"][key], color=GRAY, ls=ls, lw=0.8)
ax.set_yscale("log")
ax.text(res["rpm_resonance"]["2P_f2"] - 80, 8.0, "2P $=f_2$", ha="right", va="top", color=INK2, fontsize=7)
ax.axvline(rpm_h, color=GRAY, ls=":", lw=0.8); ax.text(rpm_h + 60, 8.0, "hover", va="top", color=INK2, fontsize=7)
ax.set_xlabel("motor speed (rpm)"); ax.set_ylabel("power per patch (µW)"); panel(ax, "c")

ax = axs[1, 1]
for i, name in enumerate(ops):
    ax.plot(R_grid / 1e3, np.array(Rsweep[name]["P"]) * 1e6, color=C[i], ls=["-", "--", "-.", ":"][i], label=name)
ax.axvline(R_des / 1e3, color=GRAY, lw=0.8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("load resistance $R$ (kΩ)"); ax.set_ylabel("power per patch (µW)")
ax.legend(loc="lower center", ncol=2); panel(ax, "d")
fig.tight_layout(); fig.savefig("figs/fig_fea.pdf"); fig.savefig("figs/fig_fea.png")

print(json.dumps({k: res[k] for k in ("verification", "modes_with_tip", "rpm_resonance", "R_design", "R_sweep", "mission")}, indent=1, default=float))
for lab, row in tab.items():
    print(lab, {k: (round(v["P_uW"], 4) if isinstance(v, dict) else round(v, 3)) for k, v in row.items()})
print(json.dumps(sens, indent=1, default=float))

"""Write paper/numbers.tex and paper/tab_*.tex directly from results/*.json (no hand-typed numbers)."""
import json, os
import numpy as np
import env as E

os.makedirs("paper", exist_ok=True)
F = json.load(open("results/fea.json"))
T = json.load(open("results/tests.json"))
D = json.load(open("results/drl.json")) if os.path.exists("results/drl.json") else None
out = []


def mac(name, val):
    out.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def f(x, n=1):
    return f"{x:.{n}f}"


def sci(x, n=1):
    m, e = f"{x:.{n}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


v = F["verification"]; d = F["derived"]; p = F["params"]
mac("fOneNoTip", f(v["f_no_tip_FE"][0], 2)); mac("fTwoNoTip", f(v["f_no_tip_FE"][1], 2))
mac("fOneNoTipAn", f(v["f1_no_tip_analytic"], 2)); mac("fOneErr", f"{v['f1_err_pct']:.1e}".replace("e-0", "\\times10^{-") + "}")
mac("fOne", f(F["modes_with_tip"][0], 2)); mac("fTwo", f(F["modes_with_tip"][1], 1)); mac("fThree", f(F["modes_with_tip"][2], 1))
mac("fOneRay", f(v["f1_tip_rayleigh"], 2))
mac("tipStatic", f(v["tip_static_FE"] * 1e3, 3))
mac("EI", f(d["EI"], 2)); mac("Cp", f(d["Cp"] * 1e9, 1)); mac("kThreeOne", f(d["k31"], 3)); mac("Yp", f(d["Yp"] / 1e9, 1))
mac("rpmHover", f"{d['rpm_hover']:,.0f}"); mac("Thover", f(d["T_hover"], 2))
mac("rpmMax", f"{p['rpm_max']:,.0f}")
mac("Rdes", f(F["R_design"] / 1e3, 1))
mac("RoptHover", f(F["R_sweep"]["hover"]["R_opt"] / 1e3, 1)); mac("RoptMax", f(F["R_sweep"]["max"]["R_opt"] / 1e3, 1))
mac("rpmRes", f"{F['rpm_resonance']['2P_f2']:,.0f}")
rs = F["rpm_sweep"]; i = int(np.argmax(rs["P_uW"]))
mac("PpeakRes", f(rs["P_uW"][i], 1)); mac("VocPeak", f(rs["Voc2_mV"][i], 0)); mac("rpmPeak", f"{rs['rpm'][i]:,.0f}")
pt = F["patch_table"]; root = pt["root (P-R)"]
mac("ProotHover", f(root["hover"]["P_uW"], 3)); mac("ProotMax", f(root["max"]["P_uW"], 2))
mac("VocRootHover", f(root["hover"]["Voc_1P_mV"], 0)); mac("VocRootMaxTwo", f(root["max"]["Voc_2P_mV"], 0))
mac("strainRootHover", f(root["hover"]["strain_1P_ue"], 2)); mac("strainRootMax", f(root["max"]["strain_2P_ue"], 2))
mac("PmisPatch", f(root["mission_avg_uW"], 3)); mac("PmidMis", f(pt["mid-span (P-M)"]["mission_avg_uW"], 3))
mac("PtipMis", f(pt["motor mount (P-T)"]["mission_avg_uW"], 3))
mac("rootOverTip", f(root["mission_avg_uW"] / pt["motor mount (P-T)"]["mission_avg_uW"], 1))
m = F["mission"]
mac("PmisFour", f(m["P_harvest_4patch_W"] * 1e6, 2)); mac("EmisFour", f(m["E_harvest_J"] * 1e3, 2))
mac("PpropMis", f(m["P_propulsion_W"], 0)); mac("EpropMis", f(m["E_propulsion_J"] / 1e3, 0))
mac("ratioMis", "$" + sci(m["ratio"]) + "$")
mac("perezRatio", "$" + sci(5.35e-3 / m["P_propulsion_W"]) + "$")
mac("ltcQ", f(450e-9 * 2.7 * 1e6, 1))
s = F["sensitivity"]
mac("PnoTip", f(s["m_tip=0.0"]["P_uW"], 2))
prof = F["profile"]
mac("profHover", f(100 * prof["hover"], 0)); mac("profCruise", f(100 * prof["cruise"], 0))
mac("profClimb", f(100 * prof["climb"], 0)); mac("profMax", f(100 * prof["max"], 0))
mac("Phov", f(E.P_HOVER, 1)); mac("nTests", str(len(T))); mac("nPass", str(sum(t["ok"] for t in T)))

# patch table
rows = []
for lab, key in [("Root (P-R)", "root (P-R)"), ("Mid-span (P-M)", "mid-span (P-M)"), ("Motor mount (P-T)", "motor mount (P-T)")]:
    r = pt[key]
    rows.append(f"{lab} & {r['x_c_mm']:.1f} & " + " & ".join(f"{r[o]['P_uW']:.3f}" for o in ("hover", "cruise", "climb", "max"))
                + f" & {r['mission_avg_uW']:.3f} \\\\")
open("paper/tab_patch.tex", "w").write("\n".join(rows) + "\n")
# operating points table
rows = []
for o, lab in (("hover", "Hover"), ("cruise", "Cruise"), ("climb", "Climb"), ("max", "Maximum")):
    r = root[o]; rpm = F["operating_points"][o]
    rows.append(f"{lab} & {rpm:,.0f} & {rpm/60:.1f} / {2*rpm/60:.1f} & {r['strain_1P_ue']:.2f} / {r['strain_2P_ue']:.2f} & "
                f"{r['Voc_1P_mV']:.0f} / {r['Voc_2P_mV']:.0f} & {F['R_sweep'][o]['R_opt']/1e3:.1f} & {r['P_uW']:.3f} \\\\")
open("paper/tab_ops.tex", "w").write("\n".join(rows) + "\n")
# sensitivity table
lab = {"m_tip": "Tip mass $m_\\mathrm{tip}$ (g)", "h": "Arm thickness $h$ (mm)", "zeta": "Damping ratio $\\zeta$",
       "delta_2P": "2P thrust ripple $\\delta_{2P}$", "delta_1P": "1P thrust asymmetry $\\delta_{1P}$"}
scale = {"m_tip": 1e3, "h": 1e3, "zeta": 1, "delta_2P": 1, "delta_1P": 1}
base = {"m_tip": p["m_tip"], "h": p["h"], "zeta": p["zeta"], "delta_2P": p["delta_2P"], "delta_1P": p["delta_1P"]}
rows = [f"Baseline (Table~\\ref{{tab:params}}) & -- & {F['modes_with_tip'][0]:.2f} & {F['modes_with_tip'][1]:.1f} & {s['baseline']:.3f} \\\\ \\midrule"]
for k in lab:
    for kk, vv in s.items():
        if kk.startswith(k + "="):
            val = float(kk.split("=")[1]) * scale[k]
            vs = f"{val:g}"
            rows.append(f"{lab[k]} & {vs} & {vv['f1']:.2f} & {vv['f2']:.1f} & {vv['P_uW']:.3f} \\\\")
open("paper/tab_sens.tex", "w").write("\n".join(rows) + "\n")
# unit tests list
def tex_name(n):
    for a, b in (("%", "\\%"), ("_", "\\_"), ("->", "$\\rightarrow$"), ("^2", "$^2$"), ("omega", "$\\omega$"), ("R + body", "$R$ + body"), ("< ", "$<$ "), ("^3", "$^3$"), (" = ", " $=$ ")):
        n = n.replace(a, b)
    return n


open("paper/tab_tests.tex", "w").write("\n".join(
    str(i + 1) + " & " + tex_name(t["name"]) + " & " + ("Pass" if t["ok"] else "Fail") + " \\\\" for i, t in enumerate(T)) + "\n")

# ------------------------------------------------------------------ DRL
if D:
    R = D["runs"]

    def pm(name, key, n=1):
        x = R[name][key]
        return f"{x['mean']:.{n}f} $\\pm$ {x['sd']:.{n}f}"

    for name, tag in [("SAC-energy", "sac"), ("PPO-energy", "ppo"), ("DQN-energy", "dqn"), ("SAC-standard", "std"), ("SAC-energy_harv", "harv")]:
        mac(f"{tag}SR", f(R[name]["SR"]["mean"], 1)); mac(f"{tag}SRsd", f(R[name]["SR"]["sd"], 1))
        mac(f"{tag}Jm", f(R[name]["J_per_m"]["mean"], 1)); mac(f"{tag}Jmsd", f(R[name]["J_per_m"]["sd"], 1))
        mac(f"{tag}Eh", f(R[name]["Eh_per_s_uW"]["mean"], 3)); mac(f"{tag}rpm", f"{R[name]['rpm']['mean']:,.0f}")
        mac(f"{tag}speed", f(R[name]["speed"]["mean"], 2)); mac(f"{tag}t", f(R[name]["t_succ"]["mean"], 1))
        mac(f"{tag}crash", f(R[name]["crash"]["mean"], 1)); mac(f"{tag}E", f(R[name]["E_succ"]["mean"], 0))
    A = D["astar"]
    mac("astarSR", f(A["SR"], 1)); mac("astarJm", f(A["J_per_m"], 1)); mac("astarSpeed", f(A["speed"], 2)); mac("astarE", f(A["E_succ"], 0))
    mac("astarEh", f(A["Eh_per_s_uW"], 3))
    sa = D["stats_algos"]
    mac("anovaF", f(sa["SR"]["anova"]["F"], 2)); mac("anovaP", f"{sa['SR']['anova']['p']:.2g}")
    mac("anovaJmF", f(sa["J_per_m"]["anova"]["F"], 2)); mac("anovaJmP", f"{sa['J_per_m']['anova']['p']:.2g}")
    mac("harvRatio", "$" + sci(D["harvest_to_propulsion_max_ratio"]) + "$")

    def fmt_p(pv):
        return "$<0.001$" if pv < 0.001 else f"{pv:.3f}"

    def stat_rows(block, metrics, names):
        rows = []
        for mname, mlab in metrics:
            for r in block[mname]["pairs"]:
                rows.append(f"{mlab} & {names[r['a']]} vs {names[r['b']]} & {(format(r['diff'], '+.4f') if abs(r['diff']) < 0.1 else format(r['diff'], '+.2f'))} & {r['t']:.2f} & {fmt_p(r['p'])} & {fmt_p(r['p_holm'])} & {r['g']:.2f} \\\\")
            if "anova" in block[mname]:
                a = block[mname]["anova"]
                rows.append(f"{mlab} & one-way ANOVA (3 groups) & -- & $F$={a['F']:.2f} & {fmt_p(a['p'])} & -- & -- \\\\")
            rows.append("\\midrule")
        return "\n".join(rows[:-1]) + "\n"

    nm = {"SAC-energy": "SAC", "PPO-energy": "PPO", "DQN-energy": "DQN",
          "SAC-standard": "Standard", "SAC-energy_harv": "Harvest-bonus"}
    nm2 = dict(nm); nm2["SAC-energy"] = "Energy-aware"
    open("paper/tab_stats_algos.tex", "w").write(stat_rows(sa, [("SR", "Success (pp)"), ("J_per_m", "Energy (J/m)")], nm))
    open("paper/tab_stats_reward.tex", "w").write(stat_rows(D["stats_reward"], [("SR", "Success (pp)"), ("J_per_m", "Energy (J/m)"),
                                                                                ("rpm", "Mean rotor speed (rpm)"), ("Eh_per_s_uW", "Harvest power ($\\mu$W)")], nm2))
    rows = []
    for name, lab in [("SAC-energy", "SAC"), ("PPO-energy", "PPO"), ("DQN-energy", "DQN")]:
        rows.append(f"{lab} (mapless, energy-aware reward) & {pm(name,'SR')} & {pm(name,'crash')} & {pm(name,'timeout')} & {pm(name,'J_per_m')} & {pm(name,'t_succ')} & {pm(name,'speed',2)} \\\\")
    rows.append(f"A*+PD (exact map, reference) & {A['SR']:.1f} & {A['crash']:.1f} & {A['timeout']:.1f} & {A['J_per_m']:.1f} & {A['t_succ']:.1f} & {A['speed']:.2f} \\\\")
    open("paper/tab_algos.tex", "w").write("\n".join(rows) + "\n")
    rows = []
    for name, lab in [("SAC-standard", "Standard (no energy term)"), ("SAC-energy", "Energy-aware"), ("SAC-energy_harv", "Energy-aware + harvest bonus")]:
        rows.append(f"{lab} & {pm(name,'SR')} & {pm(name,'J_per_m')} & {pm(name,'speed',2)} & {R[name]['rpm']['mean']:,.0f} $\\pm$ {R[name]['rpm']['sd']:,.0f} & {pm(name,'Eh_per_s_uW',3)} \\\\")
    open("paper/tab_reward.tex", "w").write("\n".join(rows) + "\n")
    rows = []
    for c, r in D["robustness"].items():
        cl = c.replace("σ", "$\\sigma$")
        rows.append(f"{cl} & {r['SAC']['mean']:.1f} $\\pm$ {r['SAC']['sd']:.1f} & {r['astar']:.1f} \\\\")
    open("paper/tab_robust.tex", "w").write("\n".join(rows) + "\n")
    rob = D["robustness"]
    mac("robNom", f(rob["nominal"]["SAC"]["mean"], 1)); mac("robNoiseTen", f(rob["ray noise σ=0.10"]["SAC"]["mean"], 1))
    mac("robWindOne", f(rob["wind σ=1.0"]["SAC"]["mean"], 1)); mac("robAstarWindOne", f(rob["wind σ=1.0"]["astar"], 1))
    mac("robComb", f(rob["noise 0.05 + wind 0.5"]["SAC"]["mean"], 1))
    mac("robWindHalf", f(rob["wind σ=0.5"]["SAC"]["mean"], 1)); mac("robAstarWindHalf", f(rob["wind σ=0.5"]["astar"], 1))
    mac("robNoiseFive", f(rob["ray noise σ=0.05"]["SAC"]["mean"], 1))
    sr = D["stats_reward"]["J_per_m"]["pairs"]; sa2 = D["stats_algos"]
    g = {(r["a"], r["b"]): r for r in sr}
    mac("pStdEn", f"{g[('SAC-standard','SAC-energy')]['p_holm']:.3f}"); mac("gStdEn", f(g[('SAC-standard','SAC-energy')]['g'], 2))
    mac("pEnHarv", f"{g[('SAC-energy','SAC-energy_harv')]['p_holm']:.3f}"); mac("gEnHarv", f(abs(g[('SAC-energy','SAC-energy_harv')]['g']), 2))
    mac("anovaRewJmF", f(D["stats_reward"]["J_per_m"]["anova"]["F"], 2)); mac("anovaRewJmP", f"{D['stats_reward']['J_per_m']['anova']['p']:.3f}")
    mac("anovaRewSRP", f"{D['stats_reward']['SR']['anova']['p']:.2f}")
    ga = {(r["a"], r["b"]): r for r in sa2["SR"]["pairs"]}
    mac("pSacPpo", f"{ga[('SAC-energy','PPO-energy')]['p_holm']:.3f}"); mac("pSacDqn", f"{ga[('SAC-energy','DQN-energy')]['p_holm']:.3f}")
    mac("pPpoDqn", f"{ga[('PPO-energy','DQN-energy')]['p_holm']:.2f}")
    mac("savePct", f(100 * (R["SAC-standard"]["J_per_m"]["mean"] - R["SAC-energy"]["J_per_m"]["mean"]) / R["SAC-standard"]["J_per_m"]["mean"], 1))
    mac("harvLossPct", f(100 * (R["SAC-energy_harv"]["J_per_m"]["mean"] - R["SAC-energy"]["J_per_m"]["mean"]) / R["SAC-energy"]["J_per_m"]["mean"], 1))
    mac("ppoLessPct", f(100 * (R["SAC-energy"]["J_per_m"]["mean"] - R["PPO-energy"]["J_per_m"]["mean"]) / R["SAC-energy"]["J_per_m"]["mean"], 1))
    mac("sacVsAstarPct", f(100 * (A["J_per_m"] - R["SAC-energy"]["J_per_m"]["mean"]) / A["J_per_m"], 1))
    mac("sacEhuJ", f(R["SAC-energy"]["Eh_uJ"]["mean"], 2))
    mac("epRatio", "$" + sci(R["SAC-energy"]["Eh_uJ"]["mean"] * 1e-6 / R["SAC-energy"]["E_succ"]["mean"]) + "$")
    mac("trajSeed", str(D["traj_example_seed"]))

open("paper/numbers.tex", "w").write("\n".join(out) + "\n")
print(len(out), "macros")

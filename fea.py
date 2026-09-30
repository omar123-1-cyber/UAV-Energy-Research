"""
fea.py -- Euler-Bernoulli finite-element model of a DJI F450 arm carrying a
motor/propeller tip mass, with a surface-bonded PZT-5A patch (d31 mode).

All quantities SI unless noted. Every assumption is collected in PARAMS so the
manuscript can list it, and every output is derived from these values only.

Electromechanical model (weak-coupling, one-way):
  * beam FE -> harmonic tip-force response via modal superposition (modal damping)
  * patch-averaged surface strain from nodal slopes:  integral of w'' = w'(x2) - w'(x1)
  * short-circuit charge amplitude  Q = d31 * Y_p * b_p * z_p * [w'(x2) - w'(x1)]
  * patch = current source (j*w*Q) in parallel with C_p, feeding resistor R
        V = j w Q R / (1 + j w R C_p),   P = |V|^2 / (2R)     (time-average)
        V_oc = Q / C_p,                  R_opt = 1 / (w C_p)
  Harmonic contributions (1P, 2P) are at different frequencies, so their
  time-averaged powers add.
"""
import numpy as np
from scipy.linalg import eigh

EPS0 = 8.854e-12

PARAMS = dict(
    # arm (equivalent solid rectangular section, geometry from the original manuscript)
    L=0.245, b=0.016, h=0.006, E=18.5e9, rho=1450.0, zeta=0.02,
    # tip mass: 2213-class motor (~56 g) + 9.4-in prop (~12 g) + mount/screws (~5 g)
    m_tip=0.073,
    # PZT-5A patch 50 x 15 x 0.2 mm
    Lp=0.050, bp=0.015, tp=0.0002, d31=-171e-12, s11E=16.4e-12, eps33T_r=1700.0,
    # propeller / vehicle
    D=0.239, CT=0.11, CP=0.045, rho_air=1.225, n_blades=2,
    mass=1.30, g=9.81, eta_prop=0.75, P_avionics=6.0,
    rpm_max=8200.0,
    # vertical dynamic tip load as a fraction of mean thrust (assumed; P scales with delta^2)
    delta_1P=0.01,   # blade-to-blade thrust asymmetry / track error at shaft frequency
    delta_2P=0.02,   # blade-passing thrust ripple (2 blades -> 2x shaft frequency)
    n_elem=80,
)


def derived(p=PARAMS):
    d = {}
    d["I"] = p["b"] * p["h"] ** 3 / 12
    d["EI"] = p["E"] * d["I"]
    d["mA"] = p["rho"] * p["b"] * p["h"]
    d["Yp"] = 1.0 / p["s11E"]
    eps33T = p["eps33T_r"] * EPS0
    d["k31"] = abs(p["d31"]) / np.sqrt(p["s11E"] * eps33T)
    eps33S = eps33T * (1 - d["k31"] ** 2)
    d["Cp"] = eps33S * p["Lp"] * p["bp"] / p["tp"]
    d["zp"] = p["h"] / 2 + p["tp"] / 2
    d["kT"] = p["CT"] * p["rho_air"] * p["D"] ** 4          # T = kT * n^2 (n in rev/s)
    d["T_hover"] = p["mass"] * p["g"] / 4
    d["rpm_hover"] = 60 * np.sqrt(d["T_hover"] / d["kT"])
    return d


def assemble(p=PARAMS):
    d = derived(p)
    n = p["n_elem"]; le = p["L"] / n; EI = d["EI"]; mA = d["mA"]
    ke = EI / le ** 3 * np.array([[12, 6 * le, -12, 6 * le],
                                  [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                  [-12, -6 * le, 12, -6 * le],
                                  [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
    me = mA * le / 420 * np.array([[156, 22 * le, 54, -13 * le],
                                   [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
                                   [54, 13 * le, 156, -22 * le],
                                   [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
    ndof = 2 * (n + 1)
    K = np.zeros((ndof, ndof)); M = np.zeros((ndof, ndof))
    for e in range(n):
        idx = slice(2 * e, 2 * e + 4)
        K[idx, idx] += ke; M[idx, idx] += me
    M[-2, -2] += p["m_tip"]
    free = np.arange(2, ndof)               # clamp w0, theta0
    return K[np.ix_(free, free)], M[np.ix_(free, free)], le, n


def modes(p=PARAMS, k=6):
    K, M, le, n = assemble(p)
    w2, phi = eigh(K, M, subset_by_index=[0, k - 1])
    return np.sqrt(w2), phi, le, n            # phi mass-normalised


def slopes_response(omega, F, p=PARAMS, nm=8, _cache={}):
    """complex nodal slope vector (nodes 0..n) for a harmonic tip force amplitude F [N]."""
    key = id(p), tuple(sorted((k, v) for k, v in p.items()))
    if key not in _cache:
        _cache.clear(); _cache[key] = modes(p, nm)
    wn, phi, le, n = _cache[key]
    tip = phi.shape[0] - 2                    # tip transverse dof in reduced system
    Hr = 1.0 / (wn ** 2 - omega ** 2 + 2j * p["zeta"] * wn * omega)
    q = phi @ (Hr * phi[tip, :] * F)          # reduced displacement vector
    theta = np.concatenate([[0.0], q[1::2]])  # nodal slopes incl. clamped node
    w = np.concatenate([[0.0], q[0::2]])
    return theta, w, le


def patch_charge(theta, le, x_c, p=PARAMS):
    d = derived(p)
    x1, x2 = x_c - p["Lp"] / 2, x_c + p["Lp"] / 2
    xs = np.arange(len(theta)) * le
    th1 = np.interp(x1, xs, theta.real) + 1j * np.interp(x1, xs, theta.imag)
    th2 = np.interp(x2, xs, theta.real) + 1j * np.interp(x2, xs, theta.imag)
    return p["d31"] * d["Yp"] * p["bp"] * d["zp"] * (th2 - th1), (th2 - th1) * d["zp"] / p["Lp"]


def harvest(rpm, x_c, R, p=PARAMS):
    """Return dict with power [W], open-circuit voltage amplitudes [V], strains, per harmonic."""
    d = derived(p)
    n_rev = rpm / 60.0
    T = d["kT"] * n_rev ** 2
    out = dict(P=0.0, Voc=[], V=[], strain=[], f=[], F=[])
    for harm, delta in ((1, p["delta_1P"]), (p["n_blades"], p["delta_2P"])):
        om = 2 * np.pi * n_rev * harm
        F = delta * T
        theta, w, le = slopes_response(om, F, p)
        Q, eps = patch_charge(theta, le, x_c, p)
        V = 1j * om * Q * R / (1 + 1j * om * R * d["Cp"])
        out["P"] += abs(V) ** 2 / (2 * R)
        out["Voc"].append(abs(Q) / d["Cp"]); out["V"].append(abs(V))
        out["strain"].append(abs(eps)); out["f"].append(om / 2 / np.pi); out["F"].append(F)
    return out


def rotor_power(rpm, p=PARAMS):
    """electrical power of ONE rotor [W] from CP model."""
    n_rev = rpm / 60.0
    return p["CP"] * p["rho_air"] * n_rev ** 3 * p["D"] ** 5 / p["eta_prop"]


def rpm_from_thrust(T, p=PARAMS):
    d = derived(p)
    return 60 * np.sqrt(np.maximum(T, 0.0) / d["kT"])


# ----------------------------------------------------------------- analytic checks
def analytic_f1_no_tip(p=PARAMS):
    d = derived(p)
    return 1.875104 ** 2 / (2 * np.pi) * np.sqrt(d["EI"] / (d["mA"] * p["L"] ** 4))


def analytic_tip_static(p=PARAMS, F=1.0):
    return F * p["L"] ** 3 / (3 * derived(p)["EI"])

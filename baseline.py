"""Classical map-based baseline: 3-D A* on an inflated occupancy grid + PD path tracking.
It has PRIVILEGED access to the exact obstacle map (the DRL agents only see 10 range rays)."""
import heapq, itertools
import numpy as np
import env as E

RES = 0.2
INFL = E.R_BODY + 0.25


def plan(sc):
    nx, ny, nz = (np.ceil(E.BOX / RES).astype(int) + 1)
    gx, gy, gz = np.meshgrid(np.arange(nx) * RES, np.arange(ny) * RES, np.arange(nz) * RES, indexing="ij")
    pts = np.stack([gx, gy, gz], -1)
    occ = (gz < E.Z_MIN + 0.2) | (gz > E.BOX[2] - 0.2) | (gx < 0.2) | (gx > E.BOX[0] - 0.2) | (gy < 0.2) | (gy > E.BOX[1] - 0.2)
    for c, r in zip(sc["C"], sc["R"]):
        occ |= np.linalg.norm(pts - c, axis=-1) < r + INFL
    s = tuple(np.round(sc["start"] / RES).astype(int)); g = tuple(np.round(sc["goal"] / RES).astype(int))
    occ[s] = False; occ[g] = False
    nb = [d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]
    cost = {d: np.sqrt(sum(abs(x) for x in d)) for d in nb}
    h = lambda a: np.sqrt(sum((a[i] - g[i]) ** 2 for i in range(3)))
    openq = [(h(s), 0.0, s)]; came = {s: None}; gs = {s: 0.0}
    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur == g:
            break
        if gc > gs[cur]:
            continue
        for d in nb:
            n = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
            if not (0 <= n[0] < nx and 0 <= n[1] < ny and 0 <= n[2] < nz) or occ[n]:
                continue
            ng = gc + cost[d]
            if ng < gs.get(n, 1e18):
                gs[n] = ng; came[n] = cur
                heapq.heappush(openq, (ng + h(n), ng, n))
    if g not in came:
        return None
    path = []; c = g
    while c is not None:
        path.append(np.array(c) * RES); c = came[c]
    return np.array(path[::-1])


def run(e, scenario_seed, kp=2.0, kd=1.5, look=1.5):   # gains tuned on validation scenarios only
    e.reset(scenario_seed)
    sc = E.make_scenario(scenario_seed)
    path = plan(sc)
    info = None
    if path is None:
        return dict(success=False, crash=False, timeout=True, E=0.0, Eh=0.0, t=0.0, path=0.0, d0=e.d0, mean_rpm=0.0, no_path=True)
    path = np.vstack([path, sc["goal"]])
    k = 0
    while True:
        # advance along path to the lookahead point
        while k < len(path) - 1 and np.linalg.norm(path[k] - e.p) < look:
            k += 1
        tgt = path[k]
        a = kp * (tgt - e.p) - kd * e.v
        _, _, term, trunc, info = e.step(np.clip(a / E.A_MAX, -1, 1))
        if term or trunc:
            return info

"""Train one (algorithm, seed, reward mode) and evaluate on the fixed test set.
usage: python3 train.py ALGO SEED MODE STEPS"""
import sys, json, time, os, pickle
import numpy as np
import env as E
from agents import SAC, PPO, DQN

TEST_SEEDS = list(range(100_000, 100_200))      # 200 held-out scenarios, shared by every method
VAL_SEEDS = list(range(200_000, 200_030))       # 30 validation scenarios for learning curves
EVAL_EVERY = 10_000
DACT = E.discrete_actions()


def policy_fn(agent, algo):
    if algo == "DQN":
        return lambda s: DACT[agent.act(s, deterministic=True)]
    return lambda s: agent.act(s, deterministic=True)


def evaluate(policy, seeds, mode="energy", noise=0.0, wind=0.0, eval_seed=0):
    e = E.DroneEnv(mode, seed=eval_seed, noise_std=noise, wind_std=wind)
    out = []
    for sd in seeds:
        s = e.reset(sd); done = False
        while not done:
            s, r, term, trunc, info = e.step(policy(s)); done = term or trunc
        out.append(info)
    return out


def summarise(infos):
    succ = np.array([i["success"] for i in infos])
    return dict(success=float(succ.mean()), crash=float(np.mean([i["crash"] for i in infos])),
                timeout=float(np.mean([i["timeout"] for i in infos])))


def train(algo, seed, mode, steps):
    env = E.DroneEnv(mode, seed=seed)
    if algo == "SAC":
        ag = SAC(E.OBS_DIM, E.ACT_DIM, seed=seed)
    elif algo == "PPO":
        ag = PPO(E.OBS_DIM, E.ACT_DIM, seed=seed)
    else:
        ag = DQN(E.OBS_DIM, len(DACT), seed=seed, total_steps=steps)
    curve = []; ep_ret = []; t0 = time.time()
    s = env.reset(); R = 0.0; step = 0
    while step < steps:
        if algo == "SAC":
            a = ag.explore_action(s); s2, r, term, trunc, info = env.step(a)
            ag.observe(s, a, r, s2, term)
        elif algo == "PPO":
            a = ag.act(s); s2, r, term, trunc, info = env.step(np.clip(a, -1, 1))
            ag.rollout_step(s, a, r, s2, term, trunc)
        else:
            k = ag.act(s); s2, r, term, trunc, info = env.step(DACT[k])
            ag.observe(s, k, r, s2, term)
        R += r; step += 1; s = s2
        if term or trunc:
            ep_ret.append(R); R = 0.0; s = env.reset()
        if step % EVAL_EVERY == 0:
            v = summarise(evaluate(policy_fn(ag, algo), VAL_SEEDS, mode))
            curve.append(dict(step=step, val_success=v["success"],
                              train_return=float(np.mean(ep_ret[-50:])) if ep_ret else None,
                              wall=time.time() - t0))
            print(f"{algo} s{seed} {mode} {step} val_SR={v['success']:.2f} ret={curve[-1]['train_return']} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)
    test = evaluate(policy_fn(ag, algo), TEST_SEEDS, mode)
    tag = f"{algo}_{mode}_s{seed}"
    os.makedirs("results/runs", exist_ok=True); os.makedirs("results/models", exist_ok=True)
    json.dump(dict(algo=algo, seed=seed, mode=mode, steps=steps, curve=curve, test=test,
                   hp=ag.hp, wall_min=(time.time() - t0) / 60), open(f"results/runs/{tag}.json", "w"), default=float)
    pickle.dump(ag.policy_state(), open(f"results/models/{tag}.pkl", "wb"))
    print("DONE", tag, summarise(test), flush=True)


if __name__ == "__main__":
    algo, seed, mode, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    train(algo, seed, mode, steps)

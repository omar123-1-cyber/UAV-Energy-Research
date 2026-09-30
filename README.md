# Arm-mounted piezoelectric harvesting and energy-aware DRL for quadrotors

Code and raw results for the paper

> S. Omar, M. Guoli. *Can Arm-Mounted Piezoelectric Harvesting Extend Quadrotor Endurance? A Physics-Consistent Finite-Element and Energy-Aware Deep Reinforcement Learning Assessment.* (submitted)

The repository contains:

| File | Purpose |
|---|---|
| `fea.py` | Euler–Bernoulli FE model of a DJI F450 arm with motor tip mass and a PZT-5A patch (d31, weak coupling), rotor thrust/power model |
| `fea_run.py` | All harvester analyses (modes, patch location, rpm and load sweeps, mission budget, sensitivity) → `results/fea.json`, `figs/fig_fea.pdf` |
| `env.py` | Mapless 3-D obstacle-avoidance task with rotor-speed-dependent propulsion power and FE harvest lookup |
| `nn.py`, `agents.py` | NumPy MLP/Adam and SAC, PPO, DQN implementations |
| `baseline.py` | Map-based A* + PD reference planner |
| `train.py` | Train and test one (algorithm, seed, reward) run → `results/runs/*.json` |
| `analyze.py` | Statistics, robustness, baseline and figures → `results/drl.json`, `figs/` |
| `test_unit.py` | 25 unit tests: FE vs closed-form solutions, electrical identities, environment physics, finite-difference checks of every gradient |
| `gen_numbers.py` | Writes every number used in the manuscript from the JSON results |

## Requirements

Python ≥ 3.9 with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`). No GPU or deep-learning framework is needed.

## Reproducing the results

```bash
python3 test_unit.py            # 25/25 tests should pass
python3 fea_run.py              # harvester model, ~5 s
# DRL: 25 runs (SAC/PPO/DQN x 5 seeds, SAC with 3 reward configurations)
cat jobs.txt | xargs -P 2 -L 1 sh -c 'python3 train.py $0 $1 $2 $3'
python3 analyze.py              # statistics, robustness, figures, ~3 min
```

With two CPU cores the full training takes about 3 hours (SAC ≈ 19 min per run, PPO ≈ 2 min, DQN ≈ 5 min). The raw per-episode test results and the trained policies of the published runs are included in `results/`, so `analyze.py` can be run directly without retraining.

## Licence

MIT (see `LICENSE`).

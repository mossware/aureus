# S. aureus Three-Level Stochastic ABM

Agent-based model of *Staphylococcus aureus* population dynamics under microgravity, antibiotic treatment, mutation cost, and biofilm formation.

---

## Model overview

Each cell carries a 7-element state vector:

| Index | Name  | Description                        |
|-------|-------|------------------------------------|
| 0     | `g`   | growth rate                        |
| 1     | `m`   | metabolic rate                     |
| 2     | `s`   | stress level                       |
| 3     | `q`   | quorum sensing signal              |
| 4     | `r`   | resistance                         |
| 5     | `pheno` | phenotype (0=S, 1=T, 2=P)       |
| 6     | `agg` | aggregation state (0=planktonic, 1=biofilm) |

### Phenotypes

- **S — Susceptible**: fast growth, high antibiotic kill rate
- **T — Tolerant**: slower growth, stress-adapted, moderate resistance
- **P — Persister**: near-dormant, survives antibiotic well

### Three model levels

**Level 1 — Intracellular dynamics**  
Stress (`s`), metabolism (`m`), quorum sensing (`q`), and growth (`g`) are updated each step based on environmental inputs (microgravity, antibiotic, oxygen, nutrients, pressure, radiation).

**Level 2 — Population-level events**  
S/T/P phenotype switching, environment-specific kill rates, antibiotic treatment, cell division, and biofilm aggregation/disaggregation.

**Level 3 — Mutation**  
At division, S and T cells accumulate resistance mutations drawn from a normal distribution. T cells have a higher mutation rate scaled by stress, and can undergo hypermutation bursts. High-`r` cells pay a growth penalty: `g_adj = g_base × (1 − KR_COST × r)`. T cells partially offset this cost via a chaperone-buffering bonus.

### Biofilm / Aggregation

Planktonic cells transition to aggregated per step with probability driven by population density, QS signal, and microgravity level. Aggregated cells:
- receive 40% reduced antibiotic kill (EPS matrix shielding)
- grow 10% slower (EPS production cost)

A small spontaneous disaggregation probability (`K_DISAGG`) prevents total lock-in.

---

## Scenarios

| Name           | Description                                                        |
|----------------|--------------------------------------------------------------------|
| `control`      | Earth baseline — normal gravity, full O₂/nutrients, no antibiotic |
| `microgravity` | Spaceflight-like — suppressed QS, S→T shift, no antibiotic        |
| `antibiotic`   | Earth gravity + moderate antibiotic course                         |
| `combined`     | Spaceflight physiology + antibiotic treatment                      |
| `mars_like`    | Low-g, low-O₂, low-N, high radiation, 0.006 atm pressure          |

### Antibiotic modes

| Mode       | A_MAX | Description                                  |
|------------|-------|----------------------------------------------|
| `none`     | 0.0   | No treatment                                 |
| `mild`     | 0.30  | Sub-inhibitory / prophylactic dose           |
| `moderate` | 0.60  | Standard treatment course                    |
| `strong`   | 0.90  | Aggressive / last-resort dosing              |

---

## Files

### `staph_sim.py` — core simulation

**Key globals**

- `PARAMS` — all tunable biological parameters (growth rates, kill rates, mutation rates, biofilm thresholds, etc.)
- `SCENARIOS` — per-scenario environment and resource settings
- `AB_MODES` — antibiotic ramp/taper profiles

**`Simulation` class**

```python
sim = Simulation("combined")   # initialise
sim.run()                      # run all steps with progress bar
sim.save_csv("output.csv")     # export time-series to CSV
```

`sim.hist` is a dict of per-step lists:  
`n`, `n_S`, `n_T`, `n_P`, `frac_S`, `frac_T`, `frac_P`, `frac_agg`, `g`, `m`, `s`, `q`, `r`, `r_var`, `a`, `N_res`, `O_res`

**Plot functions**

- `plot_single(sim)` — 12-panel dashboard + 4-panel analytical figure for one scenario
- `plot_comparison(sims)` — multi-scenario overlay plots (population, resistance, stress, growth, aggregation, antibiotic)

**CLI**

```
python staph_sim.py --scenario control
python staph_sim.py --scenario microgravity
python staph_sim.py --scenario antibiotic
python staph_sim.py --scenario combined
python staph_sim.py --scenario mars_like
python staph_sim.py --scenario all             # runs all 5 and compares
python staph_sim.py --scenario combined --animate
```

---

### `run_batch.py` — batch runner

Runs each scenario N times, saves every run's time-series to CSV, and computes per-step mean ± std across runs.

**CLI**

```
python run_batch.py                                        # all scenarios, 100 runs each
python run_batch.py --runs 20
python run_batch.py --scenarios control antibiotic --runs 50
python run_batch.py --outdir my_results --runs 10
python run_batch.py --no-plots
```

**Output layout**

```
results/
  <scenario>/
    run_001.csv          per-run full time-series
    run_002.csv
    ...
    run_NNN.csv
    mean_summary.csv     per-step mean ± std across all runs
  all_scenarios_mean.csv final-step summary, one row per scenario
  plots/
    <scenario>_overview.png
    all_scenarios_overview.png
    population_structure.png
```

**`mean_summary.csv` columns**  
`step`, then for each numeric column: `<col>_mean`, `<col>_std`

**`all_scenarios_mean.csv` columns**  
`scenario`, then for each key in `[n, frac_S, frac_T, frac_P, r, r_var, s, q, g]`: `<key>_mean`, `<key>_std`, `<key>_min`, `<key>_max`, plus `n_extinct`, `n_runs`

---

## Key parameters

| Parameter        | Default | Description                                      |
|------------------|---------|--------------------------------------------------|
| `N_CAP`          | 2000    | population carrying capacity                     |
| `DT`             | 0.25    | time step size                                   |
| `STEPS`          | 10000   | steps per simulation run                         |
| `G_S/T/P`        | 0.96 / 0.55 / 0.08 | base growth rates per phenotype   |
| `KILL_S/T/P`     | 0.45 / 0.16 / 0.02 | base antibiotic kill rates        |
| `KR_COST`        | 0.08    | resistance growth penalty coefficient            |
| `KR_T_BONUS`     | 0.05    | T cell chaperone growth offset                   |
| `K_DENS`         | 0.15    | density contribution to aggregation probability  |
| `K_QS_AGG`       | 0.10    | QS contribution to aggregation probability       |
| `K_MG_AGG`       | 0.20    | microgravity contribution to aggregation         |
| `AGG_KILL_PROTECT` | 0.40  | fraction of antibiotic kill blocked by biofilm   |
| `AGG_GROWTH_DRAG` | 0.10   | fractional growth penalty inside biofilm         |
| `K_DISAGG`       | 0.005   | spontaneous disaggregation probability per step  |
| `SIGMA_0`        | 0.010   | base mutation standard deviation                 |
| `SIGMA_T_MUL`    | 2.0     | T cell mutation rate multiplier                  |

---

## Requirements

```
numpy
matplotlib
```

Both are standard; no other dependencies needed.

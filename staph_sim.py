"""
S. aureus Three-Level Stochastic ABM — Microgravity + Antibiotic + Mutation Cost + Biofilm

Usage:
  python staph_sim.py --scenario control
  python staph_sim.py --scenario microgravity
  python staph_sim.py --scenario antibiotic
  python staph_sim.py --scenario combined
  python staph_sim.py --scenario mars_like
  python staph_sim.py --scenario all           # runs all 5 and compares
  python staph_sim.py --scenario combined --animate   # live animation
"""

import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# ansi colour helpers
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    GRAY    = "\033[90m"
    ORANGE  = "\033[38;5;208m"

def c(text, *codes): return "".join(codes) + str(text) + C.RESET

# phenotype indices
PHENO_S = 0   # susceptible – fast growth, high antibiotic kill
PHENO_T = 1   # tolerant    – slower growth, survives stress, moderate r
PHENO_P = 2   # persister   – barely divides, survives antibiotic well

# cell-state column indices: 0=g, 1=m, 2=s, 3=q, 4=r, 5=pheno, 6=agg
COL_G    = 0
COL_M    = 1
COL_S    = 2
COL_Q    = 3
COL_R    = 4
COL_PHE  = 5
COL_AGG  = 6   # 0 = planktonic, 1 = aggregated (biofilm)
AB_MODES = {
    "none": dict(
        A_MAX         = 0.0,
        ramp_start    = 9999,
        ramp_end      = 9999,
        taper_start   = 9999,
        ramp_rate     = 0.008,
        taper_rate    = 0.015,
        kill_mul      = 1.0,
        description   = "No antibiotic treatment.",
    ),
    "mild": dict(
        A_MAX         = 0.30,
        ramp_start    = 60,
        ramp_end      = 200,
        taper_start   = 200,
        ramp_rate     = 0.006,
        taper_rate    = 0.020,
        kill_mul      = 0.70,
        description   = "Sub-inhibitory / prophylactic dose. Slow ramp, quick taper.",
    ),
    "moderate": dict(
        A_MAX         = 0.60,
        ramp_start    = 40,
        ramp_end      = 300,
        taper_start   = 300,
        ramp_rate     = 0.008,
        taper_rate    = 0.015,
        kill_mul      = 1.00,
        description   = "Standard treatment course. Mirrors original model behaviour.",
    ),
    "strong": dict(
        A_MAX         = 0.90,
        ramp_start    = 20,
        ramp_end      = 400,
        taper_start   = 400,
        ramp_rate     = 0.012,
        taper_rate    = 0.010,
        kill_mul      = 1.40,
        description   = "Aggressive / last-resort dosing. Hard selection pressure.",
    ),
}

SCENARIOS = {
    "control": dict(
        description = (
            "Earth baseline. Normal gravity, full oxygen and nutrients, "
            "no antibiotic pressure. Reference condition for all comparisons."
        ),
        env = dict(
            microgravity_level = 0.0,
            O_INIT             = 0.90,
            N_INIT             = 0.80,
            PRESSURE           = 1.00,
            RADIATION          = 0.00,
        ),
        ab_mode      = "none",
        env_kill_mod = dict(S=1.00, T=1.00, P=1.00),
        growth_mod   = 1.00,
        resource = dict(
            N_RENEW   = 0.018,
            O_RENEW   = 0.018,
            N_CONSUME = 0.000025,
            O_CONSUME = 0.000020,
            bg_death  = 0.0005,
        ),
    ),

    "microgravity": dict(
        description = (
            "Low-shear / spaceflight-like physiology. Agr/QS suppressed by "
            "microgravity, pushing S→T colonisation-like shift. Oxygen and "
            "nutrients slightly reduced. No antibiotic."
        ),
        env = dict(
            microgravity_level = 0.30,
            O_INIT             = 0.80,
            N_INIT             = 0.72,
            PRESSURE           = 1.00,
            RADIATION          = 0.05,
        ),
        ab_mode      = "none",
        env_kill_mod = dict(S=0.85, T=0.80, P=0.90),
        growth_mod   = 0.95,
        resource = dict(
            N_RENEW   = 0.011,
            O_RENEW   = 0.011,
            N_CONSUME = 0.00011,
            O_CONSUME = 0.000090,
            bg_death  = 0.0012,
        ),
    ),

    "antibiotic": dict(
        description = (
            "Treatment pressure under normal gravity. Moderate antibiotic "
            "course. S cells take the brunt; T and P seed recovery. "
            "Demonstrates selection-rebound dynamics."
        ),
        env = dict(
            microgravity_level = 0.00,
            O_INIT             = 0.90,
            N_INIT             = 0.80,
            PRESSURE           = 1.00,
            RADIATION          = 0.00,
        ),
        ab_mode      = "moderate",
        env_kill_mod = dict(S=1.00, T=1.00, P=1.00),
        growth_mod   = 1.00,
        resource = dict(
            N_RENEW   = 0.018,
            O_RENEW   = 0.018,
            N_CONSUME = 0.000025,
            O_CONSUME = 0.000020,
            bg_death  = 0.0005,
        ),
    ),

    "combined": dict(
        description = (
            "Space-like stress plus antibiotic treatment. Microgravity "
            "pre-adapts the population (T enrichment, QS suppression) before "
            "the antibiotic pulse. The altered physiology changes apparent kill "
            "rates and rebound dynamics."
        ),
        env = dict(
            microgravity_level = 0.30,
            O_INIT             = 0.80,
            N_INIT             = 0.72,
            PRESSURE           = 1.00,
            RADIATION          = 0.05,
        ),
        ab_mode      = "moderate",
        env_kill_mod = dict(S=0.85, T=0.80, P=0.90),
        growth_mod   = 0.95,
        resource = dict(
            N_RENEW   = 0.011,
            O_RENEW   = 0.011,
            N_CONSUME = 0.00011,
            O_CONSUME = 0.000090,
            bg_death  = 0.0012,
        ),
    ),

    "mars_like": dict(
        description = (
            "Mars-analogue: ~0.38 g surface gravity, ~0.6% O₂ atmosphere, "
            "desiccated nutrient supply, ~50× Earth cosmic ray flux, and "
            "0.006 atm pressure. Growth is severely suppressed; persister "
            "fraction dominates as the primary survival strategy."
        ),
        env = dict(
            microgravity_level = 0.62,
            O_INIT             = 0.30,
            N_INIT             = 0.40,
            PRESSURE           = 0.006,
            RADIATION          = 0.50,
        ),
        ab_mode      = "mild",
        env_kill_mod = dict(S=0.50, T=0.35, P=0.20),
        growth_mod   = 0.50,
        resource = dict(
            N_RENEW   = 0.005,
            O_RENEW   = 0.005,
            N_CONSUME = 0.00030,
            O_CONSUME = 0.00025,
            bg_death  = 0.0025,
        ),
    ),
}

COLORS = {
    "control":      "#00FF99",
    "microgravity": "#3399FF",
    "antibiotic":   "#FF6633",
    "combined":     "#CC44FF",
    "mars_like":    "#FF9900",
}

SCENARIO_LABELS = {
    "control":      "Control  (Earth baseline)",
    "microgravity": "Microgravity  (spaceflight-like)",
    "antibiotic":   "Antibiotic  (moderate, Earth)",
    "combined":     "Combined  (spaceflight + antibiotic)",
    "mars_like":    "Mars-like  (low-g, low-O₂, radiation)",
}

PARAMS = dict(
    N_CAP       = 2000,
    DT          = 0.25,
    STEPS       = 10000,

    # Level 1 – stress dynamics
    K_MU        = 0.10,
    K_A         = 0.20,
    K_RAD       = 0.08,
    K_PRES      = 0.06,
    K_S         = 0.08,

    # Level 1 – growth rates per phenotype
    G_S         = 0.96,
    G_T         = 0.55,
    G_P         = 0.08,

    # Level 1 – metabolism
    M0          = 1.00,
    KO          = 0.30,
    KN          = 0.25,

    # Level 1 – quorum sensing
    Q_MAX       = 1.00,
    KD          = 0.40,
    DELTA_MU    = 0.35,

    # Level 2 – base antibiotic kill rates per phenotype
    KILL_S      = 0.45,
    KILL_T      = 0.16,
    KILL_P      = 0.02,

    EPS         = 0.05,
    ETA_Q       = 0.18,
    GAMMA_Q     = 0.20,

    # Level 2 – phenotype switching
    K_ST_STRESS = 0.025,
    K_ST_LQS    = 0.020,
    K_TS        = 0.02,
    K_TP        = 0.015,
    K_PT        = 0.025,
    K_PS        = 0.002,

    # Level 3 – mutation
    SIGMA_0     = 0.010,
    SIGMA_T_MUL = 2.0,
    BETA_S      = 0.60,
    R_INIT      = 0.10,

    # mutation cost: g_adj = g_base × (1 − KR_COST × r); T cells get a chaperone offset
    KR_COST     = 0.08,
    KR_T_BONUS  = 0.05,   # fractional growth bonus for T cells

    # biofilm: planktonic→aggregated driven by density, QS, and microgravity
    # aggregated cells get reduced antibiotic kill and slower growth
    K_DENS          = 0.15,
    K_QS_AGG        = 0.10,
    K_MG_AGG        = 0.20,
    AGG_KILL_PROTECT= 0.40,   # fraction of kill blocked by biofilm matrix
    AGG_GROWTH_DRAG = 0.10,   # fractional growth penalty inside biofilm
    K_DISAGG        = 0.005,  # spontaneous disaggregation probability per step

    # Resource renewal/consumption (PARAMS-level defaults; scenarios override)
    N_RENEW     = 0.007,
    O_RENEW     = 0.008,
    N_CONSUME   = 0.00020,
    O_CONSUME   = 0.00015,

    ETA_Q_DIV   = 0.12,
)


class Simulation:
    """
    Three-phenotype ABM: S (susceptible), T (tolerant), P (persister).

    Per-cell state vector: [g, m, s, q, r, pheno, agg]
    Indices:                 0  1  2  3  4  5      6
    agg: 0 = planktonic, 1 = aggregated (biofilm)
    """

    def __init__(self, scenario_name, p=PARAMS):
        self.p    = dict(p)
        self.name = scenario_name
        sc        = SCENARIOS[scenario_name]

        self.env         = sc["env"]
        self.env_kill    = sc["env_kill_mod"]
        self.growth_mod  = sc["growth_mod"]
        self.ab_profile  = AB_MODES[sc["ab_mode"]]
        self.description = sc["description"]

        self.res = dict(
            N_RENEW   = p["N_RENEW"],
            O_RENEW   = p["O_RENEW"],
            N_CONSUME = p["N_CONSUME"],
            O_CONSUME = p["O_CONSUME"],
            bg_death  = 0.001,
        )
        self.res.update(sc.get("resource", {}))

        n = 150
        # 7 columns: [g, m, s, q, r, pheno, agg]
        self.cells = np.zeros((n, 7))
        self.cells[:, COL_G]   = self.p["G_S"]
        self.cells[:, COL_M]   = self.p["M0"]
        self.cells[:, COL_S]   = 0.05
        self.cells[:, COL_Q]   = 0.20
        self.cells[:, COL_R]   = self.p["R_INIT"]
        self.cells[:, COL_PHE] = float(PHENO_S)
        self.cells[:, COL_AGG] = 0.0    # all start planktonic

        self.nutrients  = self.env["N_INIT"]
        self.oxygen     = self.env["O_INIT"]
        self.antibiotic = 0.0

        self.hist = {k: [] for k in
                     ["n", "n_S", "n_T", "n_P",
                      "frac_S", "frac_T", "frac_P",
                      "frac_agg",
                      "g", "m", "s", "q", "r", "a",
                      "N_res", "O_res", "r_var"]}

    def step(self, t):
        p   = self.p
        env = self.env
        ab  = self.ab_profile
        n   = len(self.cells)

        if n == 0:
            self._record(0)
            return

        mu        = env["microgravity_level"]
        pressure  = env["PRESSURE"]
        radiation = env["RADIATION"]

        # antibiotic ramp / taper
        if ab["ramp_start"] < t <= ab["ramp_end"]:
            self.antibiotic = min(
                ab["A_MAX"],
                self.antibiotic + ab["A_MAX"] * ab["ramp_rate"]
            )
        elif t > ab["taper_start"]:
            self.antibiotic = max(
                0.0,
                self.antibiotic - ab["A_MAX"] * ab["taper_rate"]
            )
        A = self.antibiotic
        c = self.cells

        ph   = c[:, COL_PHE].astype(int)
        is_S = ph == PHENO_S
        is_T = ph == PHENO_T
        is_P = ph == PHENO_P

        # L1: stress
        pressure_stress = max(0.0, 1.0 - pressure)
        ds = (
            p["K_MU"]  * mu
            + p["K_A"] * A
            + p["K_RAD"]  * radiation
            + p["K_PRES"] * pressure_stress
            - p["K_S"]  * c[:, COL_S]
        )
        c[:, COL_S] = np.clip(c[:, COL_S] + p["DT"] * ds, 0.0, 4.0)

        # L1: metabolism
        c[:, COL_M] = p["M0"] \
                      * (self.oxygen    / (self.oxygen    + p["KO"])) \
                      * (self.nutrients / (self.nutrients + p["KN"]))

        # L1: quorum sensing
        rho      = n / p["N_CAP"]
        q_target = (p["Q_MAX"]
                    * (rho / (rho + p["KD"]))
                    * (1.0 - p["DELTA_MU"] * mu))
        q_target += np.random.normal(0.0, 0.05, size=n)
        c[:, COL_Q]  = c[:, COL_Q] + p["DT"] * (q_target - c[:, COL_Q]) * 2.0
        c[:, COL_Q]  = np.clip(c[:, COL_Q], 0.0, 1.0)

        # L2: aggregation — driven by density, QS signal, and microgravity
        planktonic = c[:, COL_AGG] == 0
        p_agg = (p["K_DENS"]    * rho
                 + p["K_QS_AGG"] * c[:, COL_Q]
                 + p["K_MG_AGG"] * mu)
        p_agg = np.clip(p_agg * p["DT"], 0.0, 1.0)
        agg_trigger = planktonic & (np.random.random(n) < p_agg)
        c[agg_trigger, COL_AGG] = 1.0

        # spontaneous disaggregation (biofilm dispersal)
        aggregated = c[:, COL_AGG] == 1
        disagg_trigger = aggregated & (np.random.random(n) < p["K_DISAGG"] * p["DT"])
        c[disagg_trigger, COL_AGG] = 0.0

        # L1: growth
        stress_factor = 1.0 / (1.0 + 0.5 * c[:, COL_S] ** 1.5)
        g_base        = np.where(is_S, p["G_S"],
                        np.where(is_T, p["G_T"],
                                       p["G_P"]))
        c[:, COL_G] = (g_base * self.growth_mod
                       * stress_factor * c[:, COL_M]
                       * (1.0 + p["ETA_Q"] * c[:, COL_Q]))

        # mutation cost: high r slows growth; T cells offset via chaperone buffering
        r_cost = np.clip(1.0 - p["KR_COST"] * c[:, COL_R], 0.10, 1.0)
        c[:, COL_G] *= r_cost
        # T cell chaperone offset
        c[is_T, COL_G] *= (1.0 + p["KR_T_BONUS"])

        # EPS production cost: aggregated cells grow slower
        agg_mask = c[:, COL_AGG] == 1
        c[agg_mask, COL_G] *= (1.0 - p["AGG_GROWTH_DRAG"])

        c[:, COL_G] = np.clip(c[:, COL_G], 0.0, 1.0)

        # L2: phenotype switching
        low_qs    = np.clip(1.0 - c[:, COL_Q], 0.0, 1.0)
        p_ST      = np.clip(
            (p["K_ST_STRESS"] * c[:, COL_S] + p["K_ST_LQS"] * low_qs) * p["DT"],
            0.0, 1.0)
        switch_ST = is_S & (np.random.random(n) < p_ST)

        p_TS      = np.clip(p["K_TS"] * (1.0 - c[:, COL_S] / 4.0) * p["DT"], 0.0, 1.0)
        switch_TS = is_T & (np.random.random(n) < p_TS)

        p_TP      = np.clip(p["K_TP"] * c[:, COL_S] * p["DT"], 0.0, 1.0)
        switch_TP = is_T & (np.random.random(n) < p_TP)

        p_PT      = np.clip(p["K_PT"] * p["DT"], 0.0, 1.0)
        p_PS      = np.clip(p["K_PS"] * p["DT"], 0.0, 1.0)
        switch_PT = is_P & (np.random.random(n) < p_PT)
        switch_PS = is_P & (~switch_PT) & (np.random.random(n) < p_PS)

        c[switch_ST, COL_PHE] = float(PHENO_T)
        c[switch_TS, COL_PHE] = float(PHENO_S)
        c[switch_TP, COL_PHE] = float(PHENO_P)
        c[switch_PT, COL_PHE] = float(PHENO_T)
        c[switch_PS, COL_PHE] = float(PHENO_S)

        # L2: environment-specific death
        A_local   = A * np.random.uniform(0.7, 1.3, size=n)
        ph        = c[:, COL_PHE].astype(int)
        is_S      = ph == PHENO_S
        is_T      = ph == PHENO_T
        is_P      = ph == PHENO_P

        base_kill = np.where(is_S, p["KILL_S"],
                    np.where(is_T, p["KILL_T"],
                                   p["KILL_P"]))
        env_mod   = np.where(is_S, self.env_kill["S"],
                    np.where(is_T, self.env_kill["T"],
                                   self.env_kill["P"]))

        if A > 0:
            eff_kill = (base_kill
                        * ab["kill_mul"]
                        * env_mod
                        * A_local / ab["A_MAX"]) \
                       / (1.0 + c[:, COL_R]) \
                       * (1.0 - p["GAMMA_Q"] * c[:, COL_Q])

            # biofilm matrix reduces effective antibiotic kill
            agg_protect = 1.0 - p["AGG_KILL_PROTECT"] * c[:, COL_AGG]
            eff_kill   *= agg_protect

            p_death = np.clip(eff_kill * p["DT"], 0.0, 1.0)
        else:
            rad_death = 0.002 * radiation
            p_death   = np.full(n, self.res["bg_death"] + rad_death)

        alive      = np.random.random(n) < (1.0 - p_death)
        self.cells = self.cells[alive]

        n = len(self.cells)
        if n == 0:
            self._record(0)
            return
        c  = self.cells
        ph = c[:, COL_PHE].astype(int)
        is_S = ph == PHENO_S
        is_T = ph == PHENO_T

        # L2: division
        if n < p["N_CAP"]:
            nutrient_factor = self.nutrients / (self.nutrients + self.res["N_CONSUME"] * n)
            nutrient_cap    = 1.0 - (n / p["N_CAP"])

            p_div  = (c[:, COL_G] * c[:, COL_M]
                      * nutrient_factor * nutrient_cap
                      * p["DT"])
            p_div  = np.clip(p_div, 0.0, 1.0)

            div_mask = np.random.random(n) < p_div

            if div_mask.any():
                daughters = c[div_mask].copy()
                # daughters inherit parent aggregation state; ~50% reset to planktonic
                daughters[np.random.random(div_mask.sum()) < 0.5, COL_AGG] = 0.0

                d_ph      = daughters[:, COL_PHE].astype(int)

                # L3: mutation
                mutating = (d_ph == PHENO_S) | (d_ph == PHENO_T)
                is_T_mut = d_ph[mutating] == PHENO_T

                if mutating.any():
                    s_vals     = daughters[mutating, COL_S]
                    rad_factor = 1.0 + 2.0 * radiation

                    sigma_base = p["SIGMA_0"] * rad_factor * np.ones(mutating.sum())
                    sigma_base[is_T_mut] = (
                        p["SIGMA_0"] * rad_factor * p["SIGMA_T_MUL"]
                        * (1.0 + p["BETA_S"] * s_vals[is_T_mut])
                    )
                    sigma_high = 5.0 * sigma_base
                    burst_p    = np.zeros(mutating.sum())
                    burst_p[is_T_mut] = 0.02 * np.clip(s_vals[is_T_mut] / 2.0, 0.0, 1.0)
                    is_burst   = np.random.random(mutating.sum()) < burst_p
                    sigma      = sigma_base.copy()
                    sigma[is_burst] = sigma_high[is_burst]
                    mutation   = np.random.normal(0.0, sigma)
                    daughters[mutating, COL_R] = np.clip(
                        daughters[mutating, COL_R] + mutation, 0.0, 3.0)

                self.cells = np.vstack([self.cells, daughters])

        # resources
        n2             = len(self.cells)
        self.nutrients = np.clip(
            self.nutrients + self.res["N_RENEW"] - self.res["N_CONSUME"] * n2,
            0.05, self.env["N_INIT"])
        self.oxygen    = np.clip(
            self.oxygen + self.res["O_RENEW"] - self.res["O_CONSUME"] * n2,
            0.05, self.env["O_INIT"])
        self._record(len(self.cells))

    def _record(self, n):
        self.hist["n"].append(n)
        if n > 0:
            ph    = self.cells[:, COL_PHE].astype(int)
            n_S   = int(np.sum(ph == PHENO_S))
            n_T   = int(np.sum(ph == PHENO_T))
            n_P   = int(np.sum(ph == PHENO_P))
            n_agg = int(np.sum(self.cells[:, COL_AGG] == 1))
            self.hist["n_S"].append(n_S)
            self.hist["n_T"].append(n_T)
            self.hist["n_P"].append(n_P)
            self.hist["frac_S"].append(n_S / n)
            self.hist["frac_T"].append(n_T / n)
            self.hist["frac_P"].append(n_P / n)
            self.hist["frac_agg"].append(n_agg / n)
            self.hist["g"].append(float(np.mean(self.cells[:, COL_G])))
            self.hist["m"].append(float(np.mean(self.cells[:, COL_M])))
            self.hist["s"].append(float(np.mean(self.cells[:, COL_S])))
            self.hist["q"].append(float(np.mean(self.cells[:, COL_Q])))
            self.hist["r"].append(float(np.mean(self.cells[:, COL_R])))
            self.hist["r_var"].append(float(np.var(self.cells[:, COL_R])))
        else:
            for k in ["n_S", "n_T", "n_P",
                      "frac_S", "frac_T", "frac_P",
                      "frac_agg",
                      "g", "m", "s", "q", "r", "r_var"]:
                self.hist[k].append(0.0)

        self.hist["a"].append(self.antibiotic)
        self.hist["N_res"].append(self.nutrients)
        self.hist["O_res"].append(self.oxygen)

    def run(self):
        steps = self.p["STEPS"]
        bar_w = 30

        scen_col = {
            "control":      C.GREEN,
            "microgravity": C.BLUE,
            "antibiotic":   C.YELLOW,
            "combined":     C.MAGENTA,
            "mars_like":    C.ORANGE,
        }.get(self.name, C.WHITE)

        label = SCENARIO_LABELS.get(self.name, self.name)
        print(f"\n  {c('▶', scen_col, C.BOLD)}  Scenario  {c(label, scen_col, C.BOLD)}")
        print(f"     {c(self.description, C.GRAY)}")
        ab_desc = self.ab_profile["description"]
        print(f"     {c('Antibiotic mode:', C.DIM)}  {c(ab_desc, C.GRAY)}")

        t0 = time.time()
        for t in range(steps):
            self.step(t)

            pct  = (t + 1) / steps
            done = int(pct * bar_w)
            bar  = c("█" * done, scen_col) + c("░" * (bar_w - done), C.GRAY)
            eta  = (time.time() - t0) / pct * (1 - pct) if pct else 0

            # show aggregated fraction in progress bar
            n_now   = len(self.cells)
            n_agg   = int(np.sum(self.cells[:, COL_AGG] == 1)) if n_now > 0 else 0
            pct_agg = n_agg / n_now * 100 if n_now > 0 else 0

            print(f"\r     [{bar}]  {c(f'{pct*100:5.1f}%', C.WHITE)}  "
                  f"N={c(f'{n_now:4d}', C.CYAN)}  "
                  f"agg={c(f'{pct_agg:4.1f}%', C.ORANGE)}  "
                  f"ETA {c(f'{eta:.1f}s', C.GRAY)}",
                  end="", flush=True)

            if len(self.cells) == 0:
                print(f"\n     {c('✖  Extinction at step ' + str(t), C.RED, C.BOLD)}")
                break
        else:
            final_r   = self.hist["r"][-1]
            final_n   = self.hist["n"][-1]
            final_agg = self.hist["frac_agg"][-1] * 100
            elapsed   = time.time() - t0
            print(f"\n     {c('✔  Done', C.GREEN, C.BOLD)}  "
                  f"N={c(final_n, C.CYAN)}  "
                  f"⟨r⟩={c(f'{final_r:.3f}', C.MAGENTA)}  "
                  f"agg={c(f'{final_agg:.1f}%', C.ORANGE)}  "
                  f"{c(f'({elapsed:.1f}s)', C.GRAY)}")

        return self

    def save_csv(self, path: str) -> str:
        """Write per-step history to CSV.
        Columns: step, scenario, n, n_S, n_T, n_P,
                 frac_S, frac_T, frac_P, frac_agg,
                 g, m, s, q, r, r_var, a, N_res, O_res
        """
        import csv, os
        h      = self.hist
        steps  = len(h["n"])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        COLS = ["step", "scenario",
                "n", "n_S", "n_T", "n_P",
                "frac_S", "frac_T", "frac_P",
                "frac_agg",
                "g", "m", "s", "q", "r", "r_var",
                "a", "N_res", "O_res"]

        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            for i in range(steps):
                n = h["n"][i]
                w.writerow([
                    i,
                    self.name,
                    n,
                    h["n_S"][i],    h["n_T"][i],    h["n_P"][i],
                    round(h["frac_S"][i],   6),
                    round(h["frac_T"][i],   6),
                    round(h["frac_P"][i],   6),
                    round(h["frac_agg"][i], 6),
                    round(h["g"][i],     6),
                    round(h["m"][i],     6),
                    round(h["s"][i],     6),
                    round(h["q"][i],     6),
                    round(h["r"][i],     6),
                    round(h["r_var"][i], 8),
                    round(h["a"][i],     6),
                    round(h["N_res"][i], 6),
                    round(h["O_res"][i], 6),
                ])
        return path


def _dark_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0d0d0d",
        "axes.facecolor":    "#111111",
        "axes.edgecolor":    "#333333",
        "axes.labelcolor":   "#cccccc",
        "xtick.color":       "#888888",
        "ytick.color":       "#888888",
        "text.color":        "#cccccc",
        "grid.color":        "#1e1e1e",
        "grid.linewidth":    0.6,
        "legend.facecolor":  "#111111",
        "legend.edgecolor":  "#333333",
        "font.family":       "monospace",
    })


def plot_single(sim):
    _dark_style()
    h    = sim.hist
    t    = np.arange(len(h["n"]))
    col  = COLORS[sim.name]
    env  = sim.env

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        (f"S. aureus ABM  ·  {sim.name.upper()}"
         f"  (mg={env['microgravity_level']}, O₂={env['O_INIT']}, "
         f"N={env['N_INIT']}, P={env['PRESSURE']}, rad={env['RADIATION']}, "
         f"ab={SCENARIOS[sim.name]['ab_mode']})"),
        fontsize=11, color="#ffffff", y=0.99
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.56, wspace=0.40)

    # row 0 – population / resistance / antibiotic
    ax_n = fig.add_subplot(gs[0, 0])
    ax_n.stackplot(t,
                   h["n_S"], h["n_T"], h["n_P"],
                   labels=["S (susceptible)", "T (tolerant)", "P (persister)"],
                   colors=["#44FF88", "#FFAA00", "#CC44FF"], alpha=0.85)
    ax_n.set_title("Phenotype Composition N(t)", fontsize=9, color="#ffffff")
    ax_n.legend(fontsize=6, loc="upper left")
    ax_n.grid(True); ax_n.set_xlabel("Step", fontsize=7)

    ax_r = fig.add_subplot(gs[0, 1])
    ax_r.plot(t, h["r"], color="#FF44AA", lw=1.5)
    ax_r.set_title("Avg Resistance ⟨r⟩", fontsize=9, color="#FF44AA")
    ax_r.grid(True); ax_r.set_xlabel("Step", fontsize=7)

    ax_a = fig.add_subplot(gs[0, 2])
    ax_a.plot(t, h["a"], color="#FF4444", lw=1.5)
    ax_a.set_title("Antibiotic A(t)", fontsize=9, color="#FF4444")
    ax_a.grid(True); ax_a.set_xlabel("Step", fontsize=7)

    # row 1 – growth / stress / quorum
    for spec, key, title, c_col in [
        (gs[1, 0], "g", "Avg Growth ⟨g⟩",  "#44FF88"),
        (gs[1, 1], "s", "Avg Stress ⟨s⟩",   "#FFAA00"),
        (gs[1, 2], "q", "Avg Quorum ⟨q⟩",   "#44CCFF"),
    ]:
        ax = fig.add_subplot(spec)
        ax.plot(t, h[key], color=c_col, lw=1.5)
        ax.set_title(title, fontsize=9, color=c_col)
        ax.grid(True); ax.set_xlabel("Step", fontsize=7)

    # row 2 – metabolism / nutrients / oxygen
    for spec, key, title, c_col in [
        (gs[2, 0], "m",     "Avg Metabolism ⟨m⟩", "#AA88FF"),
        (gs[2, 1], "N_res", "Nutrients",            "#88FF44"),
        (gs[2, 2], "O_res", "Oxygen",               "#44FFFF"),
    ]:
        ax = fig.add_subplot(spec)
        ax.plot(t, h[key], color=c_col, lw=1.5)
        ax.set_title(title, fontsize=9, color=c_col)
        ax.grid(True); ax.set_xlabel("Step", fontsize=7)

    # row 3: aggregation fraction / resistance variance / dual r+agg
    ax_agg = fig.add_subplot(gs[3, 0])
    ax_agg.fill_between(t, h["frac_agg"], color="#FF8C00", alpha=0.70, lw=0)
    ax_agg.plot(t, h["frac_agg"], color="#FFA500", lw=1.2)
    ax_agg.set_title("Aggregated (Biofilm) Fraction", fontsize=9, color="#FFA500")
    ax_agg.set_ylim(0, 1); ax_agg.grid(True); ax_agg.set_xlabel("Step", fontsize=7)

    ax_rv = fig.add_subplot(gs[3, 1])
    ax_rv.plot(t, h["r_var"], color="#FF44AA", lw=1.2)
    ax_rv.set_title("Resistance Variance Var(r)", fontsize=9, color="#FF44AA")
    ax_rv.grid(True); ax_rv.set_xlabel("Step", fontsize=7)

    # ⟨r⟩ vs aggregation fraction (cost–benefit overlay)
    ax_cb = fig.add_subplot(gs[3, 2])
    ax_cb.plot(t, h["r"],        color="#FF44AA", lw=1.4, label="⟨r⟩")
    ax_cb2 = ax_cb.twinx()
    ax_cb2.plot(t, h["frac_agg"], color="#FFA500", lw=1.0, ls="--", label="frac_agg")
    ax_cb2.tick_params(colors="#888888"); ax_cb2.set_ylabel("frac_agg", fontsize=7, color="#FFA500")
    ax_cb.set_title("Resistance vs Aggregation", fontsize=9, color="#ffffff")
    ax_cb.grid(True); ax_cb.set_xlabel("Step", fontsize=7)
    lines1, labels1 = ax_cb.get_legend_handles_labels()
    lines2, labels2 = ax_cb2.get_legend_handles_labels()
    ax_cb.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="upper left")

    fig.tight_layout()

    # analytical figure
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 4), facecolor="#0d0d0d")
    fig2.suptitle(f"Analytical Panels  ·  {sim.name.upper()}",
                  fontsize=11, color="#ffffff")

    n_arr = np.array(h["n"], dtype=float)
    n0    = max(n_arr[0], 1)
    surv  = n_arr / n0

    ax_s = axes2[0]
    ax_s.set_facecolor("#111111"); ax_s.grid(True, color="#1e1e1e")
    ax_s.plot(t, surv, color=col, lw=1.5)
    ax_s.axhline(1.0, color="#444444", lw=0.8, ls="--")
    ax_s.set_title("Survival Fraction S(t)", fontsize=9, color=col)
    ax_s.set_xlabel("Step", fontsize=8); ax_s.set_ylabel("N / N₀", fontsize=8)

    ax_p = axes2[1]
    ax_p.set_facecolor("#111111"); ax_p.grid(True, color="#1e1e1e")
    ax_p.plot(h["s"], h["g"], color="#44FF88", lw=1.2, alpha=0.8)
    ax_p.scatter([h["s"][0]], [h["g"][0]], color="#ffffff", s=30, zorder=5, label="start")
    ax_p.scatter([h["s"][-1]], [h["g"][-1]], color="#FF4444", s=30, zorder=5, label="end")
    ax_p.set_title("⟨g⟩ vs ⟨s⟩  phase-space", fontsize=9, color="#44FF88")
    ax_p.set_xlabel("⟨s⟩ stress", fontsize=8); ax_p.set_ylabel("⟨g⟩ growth", fontsize=8)
    ax_p.legend(fontsize=7)

    ax_v = axes2[2]
    ax_v.set_facecolor("#111111"); ax_v.grid(True, color="#1e1e1e")
    ax_v.plot(t, h["r_var"], color="#FF44AA", lw=1.2)
    ax_v.set_title("Resistance Variance Var(r)", fontsize=9, color="#FF44AA")
    ax_v.set_xlabel("Step", fontsize=8); ax_v.set_ylabel("Var(r)", fontsize=8)

    # aggregation vs stress phase-space
    ax_ag = axes2[3]
    ax_ag.set_facecolor("#111111"); ax_ag.grid(True, color="#1e1e1e")
    ax_ag.plot(h["s"], h["frac_agg"], color="#FFA500", lw=1.2, alpha=0.85)
    ax_ag.scatter([h["s"][0]], [h["frac_agg"][0]], color="#ffffff", s=30, zorder=5, label="start")
    ax_ag.scatter([h["s"][-1]], [h["frac_agg"][-1]], color="#FF4444", s=30, zorder=5, label="end")
    ax_ag.set_title("Aggregation vs ⟨s⟩  phase-space", fontsize=9, color="#FFA500")
    ax_ag.set_xlabel("⟨s⟩ stress", fontsize=8)
    ax_ag.set_ylabel("frac_agg", fontsize=8)
    ax_ag.legend(fontsize=7)

    fig2.tight_layout()

    fname  = f"staph_{sim.name}.png"
    fname2 = f"staph_{sim.name}_analysis.png"
    fig.savefig(fname,  dpi=150, bbox_inches="tight")
    fig2.savefig(fname2, dpi=150, bbox_inches="tight")
    print(f"\n  {c('Saved →', C.GREEN)}  {fname}  {fname2}")
    plt.show()


def plot_comparison(sims):
    _dark_style()

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("S. aureus ABM – Multi-Scenario Comparison",
                 fontsize=14, color="#ffffff", y=0.99)

    metrics = [
        ("n",        "Population N(t)",              0),
        ("r",        "Avg Resistance ⟨r⟩",          1),
        ("s",        "Avg Stress ⟨s⟩",              2),
        ("g",        "Avg Growth ⟨g⟩",              3),
        ("frac_agg", "Aggregated Fraction (Biofilm)", 4),
        ("a",        "Antibiotic A(t)",               5),
    ]

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    for key, title, idx in metrics:
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        for sim in sims:
            h = sim.hist
            t = np.arange(len(h[key]))
            ax.plot(t, h[key], color=COLORS[sim.name], lw=1.8,
                    label=sim.name, alpha=0.9)
        ax.set_title(title, fontsize=9)
        ax.grid(True)
        ax.set_xlabel("Step", fontsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig("staph_comparison.png", dpi=150, bbox_inches="tight")

    # phenotype composition
    fig_ph, axes_ph = plt.subplots(1, len(sims), figsize=(4 * len(sims), 4),
                                   facecolor="#0d0d0d")
    if len(sims) == 1:
        axes_ph = [axes_ph]
    fig_ph.suptitle("Phenotype Composition per Scenario (S / T / P)",
                    color="#ffffff", fontsize=12)
    for ax, sim in zip(axes_ph, sims):
        h = sim.hist
        t = np.arange(len(h["n"]))
        ax.set_facecolor("#111111")
        ax.stackplot(t,
                     h["n_S"], h["n_T"], h["n_P"],
                     labels=["S", "T", "P"],
                     colors=["#44FF88", "#FFAA00", "#CC44FF"], alpha=0.85)
        ax.set_title(sim.name, fontsize=9, color=COLORS[sim.name])
        ax.grid(True, color="#1e1e1e")
        ax.set_xlabel("Step", fontsize=7)
        ax.legend(fontsize=7, loc="upper left")
    fig_ph.tight_layout()
    fig_ph.savefig("staph_phenotypes.png", dpi=150, bbox_inches="tight")

    # resistance distribution
    fig2, ax_r = plt.subplots(figsize=(9, 4), facecolor="#0d0d0d")
    ax_r.set_facecolor("#111111")
    ax_r.set_title("Final Resistance Distribution", color="#ffffff", fontsize=12)
    for sim in sims:
        if len(sim.cells) > 0:
            ax_r.hist(sim.cells[:, COL_R], bins=40, alpha=0.55,
                      color=COLORS[sim.name], label=sim.name, edgecolor="none")
    ax_r.legend(fontsize=9)
    ax_r.set_xlabel("Resistance r", color="#cccccc")
    ax_r.set_ylabel("Cell count",   color="#cccccc")
    ax_r.tick_params(colors="#888888")
    ax_r.grid(True, color="#1e1e1e")
    fig2.tight_layout()
    fig2.savefig("staph_resistance_dist.png", dpi=150, bbox_inches="tight")

    # survival fraction
    fig3, ax_sv = plt.subplots(figsize=(9, 4), facecolor="#0d0d0d")
    ax_sv.set_facecolor("#111111")
    ax_sv.set_title("Survival Fraction S(t) per Scenario", color="#ffffff", fontsize=12)
    for sim in sims:
        n_arr = np.array(sim.hist["n"], dtype=float)
        n0    = max(n_arr[0], 1)
        surv  = n_arr / n0
        t     = np.arange(len(surv))
        ax_sv.plot(t, surv, color=COLORS[sim.name], lw=1.8,
                   label=sim.name, alpha=0.9)
    ax_sv.axhline(1.0, color="#444444", lw=0.8, ls="--")
    ax_sv.legend(fontsize=9)
    ax_sv.set_xlabel("Step", color="#cccccc")
    ax_sv.set_ylabel("N / N₀", color="#cccccc")
    ax_sv.tick_params(colors="#888888")
    ax_sv.grid(True, color="#1e1e1e")
    fig3.tight_layout()
    fig3.savefig("staph_survival.png", dpi=150, bbox_inches="tight")

    # resistance variance
    fig4, ax_rv = plt.subplots(figsize=(9, 4), facecolor="#0d0d0d")
    ax_rv.set_facecolor("#111111")
    ax_rv.set_title("Resistance Variance Var(r) – Mutation Burst Detector",
                    color="#ffffff", fontsize=12)
    for sim in sims:
        t = np.arange(len(sim.hist["r_var"]))
        ax_rv.plot(t, sim.hist["r_var"], color=COLORS[sim.name], lw=1.8,
                   label=sim.name, alpha=0.9)
    ax_rv.legend(fontsize=9)
    ax_rv.set_xlabel("Step", color="#cccccc")
    ax_rv.set_ylabel("Var(r)", color="#cccccc")
    ax_rv.tick_params(colors="#888888")
    ax_rv.grid(True, color="#1e1e1e")
    fig4.tight_layout()
    fig4.savefig("staph_r_variance.png", dpi=150, bbox_inches="tight")

    # phase-space overlay
    fig5, ax_ph = plt.subplots(figsize=(7, 5), facecolor="#0d0d0d")
    ax_ph.set_facecolor("#111111")
    ax_ph.set_title("⟨g⟩ vs ⟨s⟩ Phase-space (all scenarios)",
                    color="#ffffff", fontsize=12)
    for sim in sims:
        ax_ph.plot(sim.hist["s"], sim.hist["g"],
                   color=COLORS[sim.name], lw=1.4, alpha=0.85, label=sim.name)
    ax_ph.legend(fontsize=9)
    ax_ph.set_xlabel("⟨s⟩ stress", color="#cccccc")
    ax_ph.set_ylabel("⟨g⟩ growth", color="#cccccc")
    ax_ph.tick_params(colors="#888888")
    ax_ph.grid(True, color="#1e1e1e")
    fig5.tight_layout()
    fig5.savefig("staph_phase.png", dpi=150, bbox_inches="tight")

    # aggregation phase-space overlay
    fig6, ax_agg = plt.subplots(figsize=(7, 5), facecolor="#0d0d0d")
    ax_agg.set_facecolor("#111111")
    ax_agg.set_title("Aggregation vs ⟨s⟩ Phase-space (all scenarios)",
                     color="#ffffff", fontsize=12)
    for sim in sims:
        ax_agg.plot(sim.hist["s"], sim.hist["frac_agg"],
                    color=COLORS[sim.name], lw=1.4, alpha=0.85, label=sim.name)
    ax_agg.legend(fontsize=9)
    ax_agg.set_xlabel("⟨s⟩ stress", color="#cccccc")
    ax_agg.set_ylabel("Aggregated fraction", color="#cccccc")
    ax_agg.tick_params(colors="#888888")
    ax_agg.grid(True, color="#1e1e1e")
    fig6.tight_layout()
    fig6.savefig("staph_agg_phase.png", dpi=150, bbox_inches="tight")

    saved = ["staph_comparison.png", "staph_phenotypes.png",
             "staph_resistance_dist.png", "staph_survival.png",
             "staph_r_variance.png", "staph_phase.png",
             "staph_agg_phase.png"]
    print(f"\n  {c('Saved →', C.GREEN)}  " + "  ".join(saved))
    plt.show()


def animate(scenario_name):
    _dark_style()
    p   = PARAMS
    sim = Simulation(scenario_name)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"S. aureus Live  ·  {scenario_name.upper()}",
                 fontsize=12, color="#ffffff")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_pop  = fig.add_subplot(gs[0, 0])
    ax_res  = fig.add_subplot(gs[0, 1])
    ax_agg  = fig.add_subplot(gs[0, 2])   # aggregation panel
    ax_evo  = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_ph   = fig.add_subplot(gs[1, 2])   # agg phase-space

    N_ADVANCE  = 3
    frame_data = {"step": 0}

    def init():
        for ax in [ax_pop, ax_res, ax_agg, ax_evo, ax_dist, ax_ph]:
            ax.set_facecolor("#111111"); ax.grid(True)
        return []

    def update(_frame):
        if frame_data["step"] >= p["STEPS"] or len(sim.cells) == 0:
            return []
        for _ in range(N_ADVANCE):
            sim.step(frame_data["step"])
            frame_data["step"] += 1
            if frame_data["step"] >= p["STEPS"] or len(sim.cells) == 0:
                break

        t   = np.arange(len(sim.hist["n"]))
        col = COLORS[scenario_name]

        ax_pop.clear(); ax_pop.set_facecolor("#111111"); ax_pop.grid(True)
        ax_pop.stackplot(t,
                         sim.hist["n_S"], sim.hist["n_T"], sim.hist["n_P"],
                         labels=["S", "T", "P"],
                         colors=["#44FF88", "#FFAA00", "#CC44FF"], alpha=0.85)
        ax_pop.set_title("Phenotype Composition N(t)", fontsize=9, color=col)
        ax_pop.legend(fontsize=6, loc="upper left")

        ax_res.clear(); ax_res.set_facecolor("#111111"); ax_res.grid(True)
        ax_res.plot(t, sim.hist["r"], color="#FF44AA", lw=1.5, label="⟨r⟩")
        ax_res.plot(t, sim.hist["a"], color="#FF4444", lw=1.0, ls="--", label="A")
        ax_res.set_title("Resistance vs Antibiotic", fontsize=9)
        ax_res.legend(fontsize=7)

        # aggregation panel
        ax_agg.clear(); ax_agg.set_facecolor("#111111"); ax_agg.grid(True)
        ax_agg.fill_between(t, sim.hist["frac_agg"], color="#FF8C00", alpha=0.65, lw=0)
        ax_agg.plot(t, sim.hist["frac_agg"], color="#FFA500", lw=1.2)
        ax_agg.set_ylim(0, 1)
        ax_agg.set_title("Aggregated (Biofilm) Fraction", fontsize=9, color="#FFA500")

        ax_evo.clear(); ax_evo.set_facecolor("#111111"); ax_evo.grid(True)
        ax_evo.plot(t, sim.hist["s"], color="#FFAA00", lw=1.2, label="⟨s⟩")
        ax_evo.plot(t, sim.hist["g"], color="#44FF88", lw=1.2, label="⟨g⟩")
        ax_evo.plot(t, sim.hist["q"], color="#44CCFF", lw=1.2, label="⟨q⟩")
        ax_evo.set_title("Cell States", fontsize=9)
        ax_evo.legend(fontsize=7)

        ax_dist.clear(); ax_dist.set_facecolor("#111111"); ax_dist.grid(True)
        if len(sim.cells) > 0:
            ax_dist.hist(sim.cells[:, COL_R], bins=40, color="#CC44FF",
                         edgecolor="none", alpha=0.8)
        ax_dist.set_title("Resistance Distribution", fontsize=9)
        ax_dist.set_xlabel("r", fontsize=8)

        # stress vs aggregation phase-space
        ax_ph.clear(); ax_ph.set_facecolor("#111111"); ax_ph.grid(True)
        ax_ph.plot(sim.hist["s"], sim.hist["frac_agg"],
                   color="#FFA500", lw=1.2, alpha=0.85)
        ax_ph.set_title("Aggregation vs ⟨s⟩", fontsize=9, color="#FFA500")
        ax_ph.set_xlabel("⟨s⟩ stress", fontsize=8)
        ax_ph.set_ylabel("frac_agg", fontsize=8)
        return []

    ani = FuncAnimation(
        fig, update,
        frames=p["STEPS"] // N_ADVANCE + 5,
        init_func=init, interval=30, blit=False
    )
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI BANNER
# ─────────────────────────────────────────────────────────────────────────────
def print_banner():
    lines = [
        "",
        c("  ╔══════════════════════════════════════════════════════════════════╗", C.CYAN),
        c("  ║", C.CYAN) + c("   S. aureus  Evolutionary ABM  │  3-Level Stochastic Model    ", C.WHITE, C.BOLD) + c("║", C.CYAN),
        c("  ║", C.CYAN) + c("   Microgravity + Antibiotic + Mutation Cost + Biofilm          ", C.GRAY)           + c("║", C.CYAN),
        c("  ╚══════════════════════════════════════════════════════════════════╝", C.CYAN),
        "",
        c("  Levels", C.YELLOW, C.BOLD),
        c("  ├─ L1 ", C.GRAY) + c("stress · metabolism · QS · growth", C.WHITE),
        c("  ├─ L2 ", C.GRAY) + c("S/T/P switching · env kill · ab treatment · division · biofilm/aggregation (NEW)", C.WHITE),
        c("  └─ L3 ", C.GRAY) + c("bimodal/burst mutation · mutation cost fitness penalty (NEW)",  C.WHITE),
        "",
        c("  New features", C.YELLOW, C.BOLD),
        f"  {c('★', C.MAGENTA)} Mutation Cost    g_adj = g_base × (1 − KR_COST × r)  |  KR_COST={PARAMS['KR_COST']}",
        f"  {c('★', C.ORANGE)}  Biofilm          planktonic→aggregated  |  40% kill protection  |  10% growth drag",
        f"                   K_DENS={PARAMS['K_DENS']}  K_QS_AGG={PARAMS['K_QS_AGG']}  K_MG_AGG={PARAMS['K_MG_AGG']}  K_DISAGG={PARAMS['K_DISAGG']}",
        "",
        c("  Scenarios", C.YELLOW, C.BOLD),
        f"  {c('■', C.GREEN)}   control       Earth baseline  (mg=0.0, O₂=0.9, N=0.8, P=1.0, rad=0.0)",
        f"  {c('■', C.BLUE)}   microgravity  Spaceflight-like  (mg=0.3, O₂=0.8, N=0.72, rad=0.05)",
        f"  {c('■', C.YELLOW)}  antibiotic    Earth + moderate treatment  (ab_mode=moderate)",
        f"  {c('■', C.MAGENTA)} combined      Spaceflight + antibiotic",
        f"  {c('■', C.ORANGE)}  mars_like     Low-g, low-O₂, low-N, high radiation, 0.006 atm",
        "",
    ]
    print("\n".join(lines))


def print_run_header(scenario, animate_mode):
    mode  = c("ANIMATE", C.CYAN, C.BOLD) if animate_mode else c("SIMULATE", C.GREEN, C.BOLD)
    label = "all scenarios" if scenario == "all" else SCENARIO_LABELS.get(scenario, scenario)
    print(f"  {mode}  →  {c(label, C.WHITE, C.BOLD)}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="S. aureus 3-Level ABM: Microgravity + Antibiotic + Mutation Cost + Biofilm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python staph_sim.py --scenario control
  python staph_sim.py --scenario all
  python staph_sim.py --scenario mars_like
  python staph_sim.py --scenario combined --animate
        """
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="combined",
        help="Scenario to run (default: combined)"
    )
    parser.add_argument(
        "--animate", action="store_true",
        help="Show live animation instead of static plots"
    )
    args = parser.parse_args()

    print_banner()
    print_run_header(args.scenario, args.animate)

    if args.animate:
        sc = args.scenario if args.scenario != "all" else "combined"
        animate(sc)
        return

    if args.scenario == "all":
        sims = [Simulation(name).run() for name in SCENARIOS]
        print(f"\n  {c('All scenarios complete — generating comparison plots…', C.CYAN)}")
        plot_comparison(sims)
    else:
        sim = Simulation(args.scenario).run()
        print(f"\n  {c('Generating plots…', C.CYAN)}")
        plot_single(sim)

    print(f"\n  {c('Done.', C.GREEN, C.BOLD)}\n")


if __name__ == "__main__":
    main()

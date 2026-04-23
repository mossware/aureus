"""
Batch runner for the S. aureus ABM.
Runs each scenario N times, saves per-run CSVs, and computes per-step mean ± std.

Output layout:
  results/
    <scenario>/
      run_001.csv          per-run full time-series
      ...
      run_NNN.csv
      mean_summary.csv     per-step mean ± std across all runs
    all_scenarios_mean.csv final-step summary, one row per scenario
    plots/
      <scenario>_overview.png
      all_scenarios_overview.png
      population_structure.png

Usage:
  python run_batch.py                                        # all scenarios, 100 runs each
  python run_batch.py --runs 20
  python run_batch.py --scenarios control antibiotic --runs 50
  python run_batch.py --outdir my_results --runs 10
  python run_batch.py --no-plots
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import staph_sim as sim_module
from staph_sim import (
    SCENARIOS, SCENARIO_LABELS, PARAMS, Simulation, C, c
)

# must match Simulation.save_csv column order exactly
TIMESERIES_COLS = [
    "step", "scenario",
    "n", "n_S", "n_T", "n_P",
    "frac_S", "frac_T", "frac_P",
    "g", "m", "s", "q", "r", "r_var",
    "a", "N_res", "O_res",
]

# numeric columns to average (excludes step and scenario)
NUM_COLS = [c for c in TIMESERIES_COLS if c not in ("step", "scenario")]

# ansi-to-matplotlib colour mapping
SCENARIO_COLORS = {
    "control":      "#4caf50",   # green
    "microgravity": "#2196f3",   # blue
    "antibiotic":   "#ffc107",   # amber
    "combined":     "#ab47bc",   # magenta/purple
    "mars_like":    "#ff7043",   # orange-red
}

# variables to panel in the per-scenario overview plot
PANEL_VARS = [
    ("n",      "Total cells (N)",         "#e0e0e0"),
    ("frac_S", "Fraction susceptible",     "#4caf50"),
    ("frac_T", "Fraction tolerant",        "#ffc107"),
    ("frac_P", "Fraction persister",       "#f44336"),
    ("r",      "Mean growth rate ⟨r⟩",     "#29b6f6"),
    ("s",      "Mean stress ⟨s⟩",          "#ef5350"),
    ("g",      "Mean nutrient ⟨g⟩",        "#66bb6a"),
    ("a",      "Antibiotic conc. ⟨a⟩",     "#ff8a65"),
]


def _colour_for(scenario: str) -> str:
    return {
        "control":      C.GREEN,
        "microgravity": C.BLUE,
        "antibiotic":   C.YELLOW,
        "combined":     C.MAGENTA,
        "mars_like":    C.ORANGE,
    }.get(scenario, C.WHITE)


def _progress_bar(done: int, total: int, width: int = 25, col: str = C.GREEN) -> str:
    filled = int(done / total * width)
    return c("█" * filled, col) + c("░" * (width - filled), C.GRAY)


def _silent_run(scenario_name: str) -> Simulation:
    """Run a simulation without printing progress output."""
    s = Simulation(scenario_name)
    for t in range(s.p["STEPS"]):
        s.step(t)
        if len(s.cells) == 0:
            break
    return s


def _read_mean_summary(csv_path: str) -> dict[str, np.ndarray]:
    """Read a mean_summary.csv and return a dict of col_name → 1-D array (means and stds)."""
    steps, data = [], {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(int(float(row["step"])))
            for col in NUM_COLS:
                key_m = f"{col}_mean"
                key_s = f"{col}_std"
                data.setdefault(f"{col}_mean", []).append(float(row[key_m]))
                data.setdefault(f"{col}_std",  []).append(float(row[key_s]))
    result = {"step": np.array(steps)}
    for k, v in data.items():
        result[k] = np.array(v)
    return result


def build_mean_summary(run_paths: list[str], out_path: str) -> None:
    """Read all per-run CSVs, stack them, and write a per-step mean ± std CSV.
    Output columns: step, <col>_mean, <col>_std for every numeric column.
    """
    # load all runs into a 3-D array: (runs × steps × cols)
    all_data: list[list[list[float]]] = []

    for path in run_paths:
        rows: list[list[float]] = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append([float(row[col]) for col in NUM_COLS])
        all_data.append(rows)

    # pad shorter runs (extinction) with last row so average doesn't shrink
    max_steps = max(len(r) for r in all_data)
    for run in all_data:
        while len(run) < max_steps:
            run.append(run[-1])   # repeat final row

    arr = np.array(all_data, dtype=float)   # shape: (runs, steps, num_cols)
    means = arr.mean(axis=0)                # (steps, num_cols)
    stds  = arr.std(axis=0)                 # (steps, num_cols)

    header = ["step"]
    for col in NUM_COLS:
        header += [f"{col}_mean", f"{col}_std"]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for step_idx in range(max_steps):
            row = [step_idx]
            for col_idx in range(len(NUM_COLS)):
                row.append(round(float(means[step_idx, col_idx]), 6))
                row.append(round(float(stds[step_idx, col_idx]),  6))
            w.writerow(row)


FINAL_KEYS = ["n", "frac_S", "frac_T", "frac_P", "r", "r_var", "s", "q", "g"]

def build_final_summary(scenario_stats: dict[str, dict], out_path: str) -> None:
    """Write all_scenarios_mean.csv — one row per scenario, final-step statistics.
    scenario_stats: { scenario_name: { key: (mean, std, min, max) } }
    """
    header = ["scenario"]
    for key in FINAL_KEYS:
        header += [f"{key}_mean", f"{key}_std", f"{key}_min", f"{key}_max"]
    header += ["n_extinct", "n_runs"]

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for scenario, stats in scenario_stats.items():
            row = [scenario]
            for key in FINAL_KEYS:
                mean_, std_, min_, max_ = stats[key]
                row += [round(mean_, 6), round(std_, 6),
                        round(min_,  6), round(max_,  6)]
            row += [stats["n_extinct"], stats["n_runs"]]
            w.writerow(row)


def plot_results(scenarios: list[str], out_dir: str) -> None:
    """Generate per-scenario panel plots and a combined overlay from mean_summary.csv files."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive, safe for servers
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print(f"  {c('⚠  matplotlib not found — skipping plots.', C.YELLOW)}"
              f"  Install with: pip install matplotlib")
        return

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # dark style
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#c9d1d9",
        "axes.titlecolor":   "#e6edf3",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "grid.color":        "#21262d",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "text.color":        "#c9d1d9",
        "font.family":       "monospace",
        "legend.facecolor":  "#161b22",
        "legend.edgecolor":  "#30363d",
        "legend.labelcolor": "#c9d1d9",
    })

    # Load all mean summaries upfront
    summaries: dict[str, dict] = {}
    for scenario in scenarios:
        csv_path = os.path.join(out_dir, scenario, "mean_summary.csv")
        if not os.path.exists(csv_path):
            print(f"  {c(f'⚠  {csv_path} not found — skipping plot for {scenario}', C.YELLOW)}")
            continue
        summaries[scenario] = _read_mean_summary(csv_path)

    if not summaries:
        return

    n_panels = len(PANEL_VARS)
    n_cols   = 2
    n_rows   = (n_panels + 1) // n_cols   # ceiling division

    # 1. per-scenario overview plots
    print(f"\n  {c('↳ Generating per-scenario overview plots …', C.DIM)}")
    for scenario, data in summaries.items():
        col_hex = SCENARIO_COLORS.get(scenario, "#ffffff")
        label   = SCENARIO_LABELS.get(scenario, scenario)
        steps   = data["step"]

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(14, 3.5 * n_rows),
            constrained_layout=True,
        )
        fig.suptitle(
            f"S. aureus ABM — {label}  (mean ± 1 σ)",
            fontsize=14, fontweight="bold", color="#e6edf3", y=1.01,
        )
        axes_flat = axes.flatten()

        for ax_idx, (var, var_label, var_colour) in enumerate(PANEL_VARS):
            ax = axes_flat[ax_idx]
            mean_key = f"{var}_mean"
            std_key  = f"{var}_std"

            if mean_key not in data:
                ax.set_visible(False)
                continue

            mean_arr = data[mean_key]
            std_arr  = data[std_key]

            ax.plot(steps, mean_arr, color=var_colour, linewidth=1.5,
                    label="mean")
            ax.fill_between(
                steps,
                mean_arr - std_arr,
                mean_arr + std_arr,
                color=var_colour, alpha=0.18, linewidth=0,
                label="±1 σ",
            )
            ax.set_title(var_label, fontsize=10, pad=6)
            ax.set_xlabel("Step", fontsize=8)
            ax.grid(True)
            ax.tick_params(labelsize=7)

            # colour spine with scenario colour for quick visual ID
            for spine in ax.spines.values():
                spine.set_edgecolor(col_hex)
                spine.set_linewidth(1.2)

        # hide unused axes
        for ax_idx in range(len(PANEL_VARS), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        out_path = os.path.join(plots_dir, f"{scenario}_overview.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"     {c('✔', C.GREEN)}  {c(scenario + '_overview.png', col_hex)}")

    # 2. combined overlay plot
    print(f"\n  {c('↳ Generating combined overlay plot …', C.DIM)}")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 3.5 * n_rows),
        constrained_layout=True,
    )
    fig.suptitle(
        "S. aureus ABM — All scenarios  (mean ± 1 σ)",
        fontsize=14, fontweight="bold", color="#e6edf3",
    )
    axes_flat = axes.flatten()

    for ax_idx, (var, var_label, _) in enumerate(PANEL_VARS):
        ax = axes_flat[ax_idx]
        mean_key = f"{var}_mean"
        std_key  = f"{var}_std"
        ax.set_title(var_label, fontsize=10, pad=6)
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True)
        ax.tick_params(labelsize=7)

        for scenario, data in summaries.items():
            if mean_key not in data:
                continue
            col_hex = SCENARIO_COLORS.get(scenario, "#ffffff")
            label   = SCENARIO_LABELS.get(scenario, scenario)
            steps   = data["step"]
            mean_arr = data[mean_key]
            std_arr  = data[std_key]

            ax.plot(steps, mean_arr, color=col_hex, linewidth=1.6,
                    label=label)
            ax.fill_between(
                steps,
                mean_arr - std_arr,
                mean_arr + std_arr,
                color=col_hex, alpha=0.10, linewidth=0,
            )

        # legend only on first panel
        if ax_idx == 0:
            ax.legend(fontsize=7, framealpha=0.7, loc="upper right")

    # hide unused
    for ax_idx in range(len(PANEL_VARS), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    out_path = os.path.join(plots_dir, "all_scenarios_overview.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"     {c('✔', C.GREEN)}  {c('all_scenarios_overview.png', C.WHITE)}")

    # 3. population structure stacked area
    print(f"\n  {c('↳ Generating population-structure stacked-area plots …', C.DIM)}")
    n_scen = len(summaries)
    fig2, axes2 = plt.subplots(
        1, n_scen,
        figsize=(5 * n_scen, 4),
        constrained_layout=True,
    )
    if n_scen == 1:
        axes2 = [axes2]

    fig2.suptitle(
        "S. aureus ABM — Population structure by scenario  (mean fractions)",
        fontsize=12, fontweight="bold", color="#e6edf3",
    )

    STRUCT_COLOURS = {
        "frac_S": "#4caf50",   # susceptible → green
        "frac_T": "#ffc107",   # tolerant    → amber
        "frac_P": "#f44336",   # persister   → red
    }
    STRUCT_LABELS = {
        "frac_S": "Susceptible",
        "frac_T": "Tolerant",
        "frac_P": "Persister",
    }

    for ax, (scenario, data) in zip(axes2, summaries.items()):
        col_hex = SCENARIO_COLORS.get(scenario, "#ffffff")
        label   = SCENARIO_LABELS.get(scenario, scenario)
        steps   = data["step"]

        fracs   = [data.get(f"{k}_mean", np.zeros_like(steps))
                   for k in ("frac_S", "frac_T", "frac_P")]
        colours = [STRUCT_COLOURS[k] for k in ("frac_S", "frac_T", "frac_P")]
        lbls    = [STRUCT_LABELS[k]  for k in ("frac_S", "frac_T", "frac_P")]

        ax.stackplot(steps, fracs, colors=colours, labels=lbls, alpha=0.80)
        ax.set_title(label, fontsize=10, color=col_hex, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Fraction", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.6)

        for spine in ax.spines.values():
            spine.set_edgecolor(col_hex)
            spine.set_linewidth(1.2)

    out_path2 = os.path.join(plots_dir, "population_structure.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"     {c('✔', C.GREEN)}  {c('population_structure.png', C.WHITE)}")

    print(f"\n  {c('All plots saved to:', C.GREEN)} {c(plots_dir, C.CYAN)}")


def run_scenario_batch(scenario_name: str, n_runs: int, out_dir: str) -> dict:
    """Run scenario_name n_runs times.
    Saves out_dir/<scenario>/run_NNN.csv and mean_summary.csv.
    Returns a dict of final-step statistics for the global summary.
    """
    col   = _colour_for(scenario_name)
    label = SCENARIO_LABELS.get(scenario_name, scenario_name)
    scen_dir = os.path.join(out_dir, scenario_name)
    os.makedirs(scen_dir, exist_ok=True)

    print(f"\n  {c('●', col, C.BOLD)}  {c(label, col, C.BOLD)}"
          f"  {c(f'({n_runs} runs)', C.GRAY)}")

    run_paths:   list[str]         = []
    final_vals:  dict[str, list]   = {k: [] for k in FINAL_KEYS}
    n_extinct    = 0
    t_start      = time.time()

    for run_idx in range(1, n_runs + 1):
        s = _silent_run(scenario_name)

        # save this run's full time-series
        csv_path = os.path.join(scen_dir, f"run_{run_idx:03d}.csv")
        s.save_csv(csv_path)
        run_paths.append(csv_path)

        # collect final-step values
        h = s.hist
        n_final = h["n"][-1]
        if n_final == 0:
            n_extinct += 1

        for key in FINAL_KEYS:
            if key in h:
                final_vals[key].append(h[key][-1])
            else:
                raise KeyError(
                    f"Key '{key}' missing from sim.hist for scenario "
                    f"'{scenario_name}' run {run_idx}. "
                    f"Available keys: {list(h.keys())}"
                )

        # inline progress
        pct  = run_idx / n_runs
        eta  = (time.time() - t_start) / pct * (1 - pct) if pct > 0 else 0
        bar  = _progress_bar(run_idx, n_runs, col=col)
        print(f"\r     [{bar}]  {c(f'{run_idx}/{n_runs}', C.WHITE)}"
              f"  N_final={c(f'{n_final:4d}', C.CYAN)}"
              f"  extinct={c(n_extinct, C.RED)}"
              f"  ETA {c(f'{eta:.0f}s', C.GRAY)}   ",
              end="", flush=True)

    elapsed = time.time() - t_start
    print(f"\n     {c('✔ Done', C.GREEN, C.BOLD)}"
          f"  {c(f'{elapsed:.1f}s total', C.GRAY)}")

    # per-step mean summary
    summary_path = os.path.join(scen_dir, "mean_summary.csv")
    print(f"     {c('↳ building mean_summary.csv …', C.DIM)}", end="", flush=True)
    build_mean_summary(run_paths, summary_path)
    print(f"\r     {c('↳ mean_summary.csv written', C.GREEN)}          ")

    # aggregate stats for global summary
    stats: dict[str, tuple] = {}
    for key in FINAL_KEYS:
        arr = np.array(final_vals[key], dtype=float)
        stats[key] = (arr.mean(), arr.std(), arr.min(), arr.max())
    stats["n_extinct"] = n_extinct
    stats["n_runs"]    = n_runs

    # compact per-key summary line
    r_mean, r_std = stats["r"][0], stats["r"][1]
    n_mean, n_std = stats["n"][0], stats["n"][1]
    print(f"     final N  {c(f'{n_mean:.0f} ± {n_std:.0f}', C.CYAN)}"
          f"   ⟨r⟩  {c(f'{r_mean:.4f} ± {r_std:.4f}', C.MAGENTA)}"
          f"   extinct  {c(n_extinct, C.RED)}/{n_runs}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for the S. aureus ABM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run_batch.py
  python run_batch.py --runs 20
  python run_batch.py --scenarios control antibiotic --runs 50
  python run_batch.py --outdir my_results --runs 10
  python run_batch.py --no-plots
        """
    )
    parser.add_argument(
        "--scenarios", nargs="+",
        choices=list(SCENARIOS.keys()),
        default=list(SCENARIOS.keys()),
        help="Scenarios to run (default: all)"
    )
    parser.add_argument(
        "--runs", type=int, default=100,
        help="Number of replicate runs per scenario (default: 100)"
    )
    parser.add_argument(
        "--outdir", default="results",
        help="Root output directory (default: results/)"
    )
    parser.add_argument(
        "--no-plots", dest="plots", action="store_false",
        help="Skip the plotting step"
    )
    parser.set_defaults(plots=True)
    args = parser.parse_args()

    # banner
    print()
    print(c("  ╔══════════════════════════════════════════════════════════╗", C.CYAN))
    print(c("  ║", C.CYAN)
          + c("   S. aureus ABM  │  Batch Runner                         ", C.WHITE, C.BOLD)
          + c("║", C.CYAN))
    print(c("  ╚══════════════════════════════════════════════════════════╝", C.CYAN))
    print()
    print(f"  {c('Scenarios :', C.YELLOW)}  {', '.join(args.scenarios)}")
    print(f"  {c('Runs each :', C.YELLOW)}  {args.runs}")
    print(f"  {c('Output    :', C.YELLOW)}  {os.path.abspath(args.outdir)}/")
    print(f"  {c('Plots     :', C.YELLOW)}  {'yes' if args.plots else 'no (--no-plots)'}")
    total_runs = len(args.scenarios) * args.runs
    est_min = total_runs * 3 / 60
    print(f"  {c('Est. time :', C.YELLOW)}  ~{est_min:.0f} min  "
          f"{c(f'({total_runs} total runs)', C.GRAY)}")
    print()

    os.makedirs(args.outdir, exist_ok=True)

    # run each scenario
    all_stats: dict[str, dict] = {}
    t_global = time.time()

    for scenario in args.scenarios:
        all_stats[scenario] = run_scenario_batch(scenario, args.runs, args.outdir)

    # global final-step summary
    global_summary_path = os.path.join(args.outdir, "all_scenarios_mean.csv")
    build_final_summary(all_stats, global_summary_path)

    # print global summary table
    elapsed_total = time.time() - t_global
    print()
    print(c("  ── Final-step summary (mean ± std across all runs) ──────────────", C.CYAN))
    print()
    header = (f"  {'Scenario':<18}  {'N_final':>14}  "
              f"{'⟨r⟩':>14}  {'frac_T':>14}  {'extinct':>8}")
    print(c(header, C.GRAY))
    print(c("  " + "─" * 76, C.GRAY))

    for scenario in args.scenarios:
        col    = _colour_for(scenario)
        stats  = all_stats[scenario]
        n_m, n_s   = stats["n"][0],      stats["n"][1]
        r_m, r_s   = stats["r"][0],      stats["r"][1]
        ft_m, ft_s = stats["frac_T"][0], stats["frac_T"][1]
        ext        = stats["n_extinct"]
        n_runs     = stats["n_runs"]
        print(f"  {c(scenario, col):<28}  "
              f"{c(f'{n_m:6.0f} ± {n_s:5.0f}', C.CYAN):>14}  "
              f"{c(f'{r_m:.4f} ± {r_s:.4f}', C.MAGENTA):>14}  "
              f"{c(f'{ft_m:.3f} ± {ft_s:.3f}', C.YELLOW):>14}  "
              f"{c(f'{ext}/{n_runs}', C.RED):>8}")

    print()
    print(f"  {c('Total time:', C.GRAY)}  {elapsed_total:.0f}s")
    print()
    print(c("  Saved:", C.GREEN))
    for scenario in args.scenarios:
        print(f"    {c(scenario+'/', _colour_for(scenario))}"
              f"  run_001…{args.runs:03d}.csv  +  mean_summary.csv")
    print(f"    {c('all_scenarios_mean.csv', C.WHITE)}")

    # ── plots ─────────────────────────────────────────────────────────────────
    if args.plots:
        print()
        print(c("  ── Plotting ─────────────────────────────────────────────────────", C.CYAN))
        plot_results(args.scenarios, args.outdir)
        print(f"    {c('plots/', C.CYAN)}  *_overview.png  +  population_structure.png")

    print()


if __name__ == "__main__":
    main()

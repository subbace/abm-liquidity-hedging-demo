"""Small synthetic demonstration using byte-preserved thesis components.

This is an engineering/methodology demonstration, not the thesis full study.
No network calls, real data, learned models, training or parameter search.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from collections.abc import Mapping
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.abm_simulator import ABMSimulator
from src.causal_black_scholes_baseline import (
    adapt_abm_v042_frame,
    compute_causal_baseline_metrics,
    simulate_causal_black_scholes_window,
)
from src.causal_hedging_data import build_causal_option_windows

ROOT = Path(__file__).resolve().parent
SEED = 20261105  # Public illustration seed, not a reserved 93xx research seed.
WINDOW_STARTS = (63, 83, 103, 123)
PREFIX_ROWS = 187
ROLE = "PUBLIC_REDUCED_DEMONSTRATION"
DISCLAIMER = (
    "Synthetic engineering demonstration, not a full thesis replication. "
    "Four overlapping windows per scenario are descriptive, not independent "
    "samples or an estimate of economic effectiveness. No neural comparison, "
    "training, real-option data, market execution or profitability claim."
)


def plain_json(value):
    """Normalize readonly mappings before serialization, without changing values."""
    if isinstance(value, Mapping):
        return {str(k): plain_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [plain_json(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scenario_parameters():
    """Two existing thesis configurations, with a new disclosed demo seed."""
    common = dict(
        T=5000, P0=100.0, P_star=100.0, x0=0.5, phi=0.8, chi=1.2,
        eta=0.001, herding_bias=0.0, seed=SEED,
        liquidity_persistence=0.90, noise_liquidity_sensitivity=0.35,
        max_noise_multiplier=3.0, stress_demand_scale=0.08,
        stress_return_scale=0.015, stress_impact_on_liquidity=0.45,
        max_stress=2.5, max_abs_return=0.10,
    )
    return {
        "baseline_stable": dict(common, sigma_N=0.02, lambda_price=0.05,
                                L0=1.0, gamma=2.0, liquidity_floor=0.10,
                                h=0.02, beta=0.5),
        "low_liquidity": dict(common, sigma_N=0.025, lambda_price=0.065,
                              L0=0.70, gamma=3.2, liquidity_floor=0.08,
                              h=0.03, beta=0.6),
    }


def causal_windows(frame, scenario):
    """Retain the original 5001-row adapter guard, then use a fixed prefix.

    Selecting the first 187 observations is declared before running. It is not
    a search for favourable windows. The four starts and 63-step maturity are
    the original causal contract; no guard or source constant is relaxed.
    """
    adapted = adapt_abm_v042_frame(frame)
    windows = build_causal_option_windows(
        adapted.iloc[:PREFIX_ROWS].copy(), path_id="public_demo__" + scenario,
        path_seed=SEED, regime=scenario, split="validation",
        risk_free_rate=0.02,
        transaction_cost_convention_id="public_demo_c02_cost_0.001",
    )
    if tuple(w.start_index for w in windows) != WINDOW_STARTS:
        raise ValueError("The predeclared demo window schedule changed")
    return windows


def verify_preserved_sources():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["preserved_sources"]:
        observed = hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest()
        if observed != entry["sha256"]:
            raise ValueError("Preserved source hash changed: " + entry["path"])
    return manifest


def run(output_dir):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    provenance = verify_preserved_sources()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures, axes = plt.subplots(2, 2, figsize=(10, 6), sharex="col")
    summaries, window_rows, cash_rows, diagnostic_rows = [], [], [], []
    for column, (scenario, parameters) in enumerate(scenario_parameters().items()):
        frame = ABMSimulator(**parameters).run()
        windows = causal_windows(frame, scenario)
        results = tuple(simulate_causal_black_scholes_window(w) for w in windows)
        summaries.append(dict(scenario=scenario, **compute_causal_baseline_metrics(results)))
        for result in results:
            window_rows.append({
                "scenario": scenario, "window_id": result.window_id,
                "start_index": result.start_index, "end_index": result.end_index,
                "inception_volatility": result.inception_volatility,
                "premium": result.premium, "hedging_error": result.hedging_error,
                "loss": result.loss, "transaction_cost": result.total_transaction_cost,
                "unit_turnover": result.unit_turnover,
                "notional_turnover": result.notional_turnover,
            })
            for j in range(63):
                cash_rows.append({
                    "window_id": result.window_id, "decision_index": j,
                    "source_index": result.start_index + j,
                    "price": result.prices[j], "delta": result.deltas[j],
                    "trade": result.trade_sizes[j], "fee": result.transaction_costs[j],
                    "cash_before": result.cash_before_trade[j],
                    "cash_after_trade": result.cash_after_trade[j],
                    "cash_after_accrual": result.cash_after_accrual[j],
                })
        # Store a small, explicitly subsampled diagnostic dataset. All 5001
        # observations are simulated; only the predetermined prefix is hedged.
        diagnostic = frame.iloc[::20][["time", "price", "liquidity", "stress_state"]].copy()
        diagnostic.insert(0, "scenario", scenario)
        diagnostic_rows.append(diagnostic)
        axes[0, column].plot(frame["time"], frame["price"], linewidth=0.8, color="#173e59")
        axes[1, column].plot(frame["time"], frame["liquidity"], linewidth=0.8, color="#006c60")
        for row in (0, 1):
            axes[row, column].axvspan(63, 186, alpha=0.15, color="#bd7a28")
            axes[row, column].grid(alpha=0.15)
        axes[0, column].set_title(scenario.replace("_", " "))
        axes[1, column].set_xlabel("Synthetic step")
    axes[0, 0].set_ylabel("Synthetic underlying price")
    axes[1, 0].set_ylabel("Model liquidity")
    figures.suptitle("Synthetic ABM paths and liquidity - public reduced demonstration", fontsize=12)
    figures.text(0.5, 0.02, "Shaded: predeclared hedging support 63-186. Scenarios share one demo seed; no performance ranking.", ha="center", fontsize=8)
    figures.tight_layout(rect=(0, 0.05, 1, 0.94))
    figures.savefig(output_dir / "synthetic_paths.png", dpi=150)
    plt.close(figures)
    pd.DataFrame(window_rows).to_csv(output_dir / "window_results.csv", index=False)
    pd.DataFrame(cash_rows).to_csv(output_dir / "accounting_trace.csv", index=False)
    pd.concat(diagnostic_rows, ignore_index=True).to_csv(output_dir / "path_diagnostics_subsampled.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False)
    report = {
        "role": ROLE, "disclaimer": DISCLAIMER,
        "source_revision": provenance["source_revision"],
        "seed": SEED, "scenarios": scenario_parameters(),
        "simulated_rows_per_scenario": 5001, "window_starts": WINDOW_STARTS,
        "maturity_steps": 63, "windows_per_scenario": 4,
        "risk_free_rate": 0.02, "steps_per_year": 252,
        "transaction_cost_rate": 0.001, "initial_trade_charged": True,
        "terminal_liquidation": False,
        "sample_dependence": "Overlapping windows; two scenarios share the same seed.",
        "metrics": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(plain_json(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    receipt = {
        "python": platform.python_version(), "platform": platform.system(),
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")},
        "wall_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "source_integrity": "PASS", "mode": ROLE,
        "files": [{"path": p.name, "bytes": p.stat().st_size,
                   "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "run_receipt.json"],
    }
    (output_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"role": ROLE, "scenarios": 2, "windows": 8,
                      "wall_seconds": receipt["wall_seconds"], "cpu_seconds": receipt["cpu_seconds"]}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()

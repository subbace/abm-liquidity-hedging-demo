"""Compare demo metrics with recorded reference values, allowing float drift."""
import argparse
import json
import math
from pathlib import Path


def compare(actual, expected):
    for field in ("role", "seed", "window_starts", "maturity_steps", "windows_per_scenario",
                  "transaction_cost_rate", "initial_trade_charged", "terminal_liquidation"):
        if actual[field] != expected[field]:
            raise ValueError("Changed demonstration contract: " + field)
    a = {row["scenario"]: row for row in actual["metrics"]}
    e = {row["scenario"]: row for row in expected["metrics"]}
    if set(a) != set(e):
        raise ValueError("Scenario identities differ")
    for scenario, reference in e.items():
        if set(a[scenario]) != set(reference):
            raise ValueError("Metric fields differ: " + scenario)
        for field, value in reference.items():
            observed = a[scenario][field]
            if field in ("scenario", "count"):
                if observed != value:
                    raise ValueError("Metric identity differs: " + scenario + "/" + field)
            elif not math.isclose(float(observed), float(value), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Metric exceeds tolerance: " + scenario + "/" + field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, default=Path("outputs/summary.json"))
    parser.add_argument("--expected", type=Path, default=Path("examples/expected_summary.json"))
    args = parser.parse_args()
    compare(json.loads(args.actual.read_text(encoding="utf-8")),
            json.loads(args.expected.read_text(encoding="utf-8")))
    print("PASS: schedule and scenario identities match; metrics within atol=rtol=1e-9")


if __name__ == "__main__":
    main()

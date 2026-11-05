# Validation record

## Original scientific environment

Executed locally on Windows on 5 November 2026 using the project's canonical
Python 3.9.1 interpreter. Installed distribution versions were NumPy 2.0.2,
pandas 2.3.3, SciPy 1.13.1 and Matplotlib 3.9.4. No environment was reinstalled
for these local checks. No real data, trained weights or training were used.

| Execution | Exit code | Result | Total wall seconds |
| --- | --- | --- | --- |
| `python -B -m unittest discover -s tests -v` | 1 | 7 PASS, 1 fixture FAIL | 11.946702 |
| `python -B -m unittest discover -s tests -k future -v` after fixture correction | 0 | Corrected case PASS | 3.898483 |
| `python -B run_demo.py --output-dir outputs` | 0 | Two paths, eight windows and outputs generated | 3.862777 |
| `python -B compare_outputs.py --actual outputs/summary.json` | 0 | Reference consistency PASS | Not measured separately |

The wall times above include interpreter startup and imports. The demonstration's
own measured `run()` section took 1.195801 wall seconds and 1.1875 CPU seconds;
those measurements exclude the imports that precede `run()`. Do not interpret
the difference as a speed comparison. Child CPU time for the unittest harness
was not measured. All measurements concern this computer, not a performance
guarantee for other laptops.

## Retained failure and correction

The original future-mutation test changed future prices upward and required a
different final hedging error. That assumption was invalid for this fixture:
the future hedge could already be saturated, leaving the final error identical.
The test correctly preserved the prior decisions and inception volatility, but
failed on that extra final-outcome assertion.

The fixture now uses a disclosed downward future-price shock and verifies that
the first affected delta changes, while every preceding delta and cash value,
the inception volatility and premium remain exactly unchanged. The source
modules, numerical tolerances and causal prefix assertions were not changed.
Only the previously failed case was rerun locally. Seven unchanged tests were
not rerun merely to replace the original failure with a green report.

Machine-readable commands, exit codes and logs are in `validation/`. The public
failing traceback replaces the local machine directory with `<demo-root>`;
the original diagnostic is preserved locally under ignored
`validation/local_only/` and is excluded from publication. The failed assertion
and its numerical values remain visible in the public log.

## Generated reference and numerical comparison

`examples/expected_summary.json` was copied from the executed demonstration's
`outputs/summary.json`; it is a recorded observation, not an invented expected
result. Both scenario mean hedging errors in that small example are negative.
No result was selected or dropped by sign. The PNG in `examples/` was generated
by the runner and visually inspected: axis labels, scenario titles, shaded
support and explanatory caption are readable.

The comparison checks fixed identities exactly and numeric metrics with
`atol=rtol=1e-9`. The original numerical modules retain their own `1e-12` checks.
The initial comparison is an output/reference consistency check in the same
environment, not an independent reproduction. Byte equality of plots, JSON
or CSV across systems is not promised.

## Clean checkout validation

At this handoff, installation and execution in a fresh independent environment
are **NOT VERIFIED**. The publication controller will perform them separately
and append the actual environment, commands, results and failures before release.
A different supported Python version used for that check must be identified as
the public demonstration's portability check, not as a replacement of the
canonical Python 3.9.1 scientific environment.

The original thesis full experiments and their statistical conclusions are not
revalidated by this package. Four overlapping windows per scenario and two
scenarios sharing a seed cannot establish independent tail-risk inference,
economic superiority, market execution feasibility or profitability.


## Clean independent environment validation

A fresh local Git clone was verified byte-for-byte and the README commands were executed in a new isolated **CPython 3.12.14 / Windows / CPU** environment. Installation, checks and examples all exited **0**. The exact commands, versions, installed distributions and wall times are in [validation/CLEAN_ENVIRONMENT.json](validation/CLEAN_ENVIRONMENT.json). Other operating systems are not tested. Timing is engineering evidence, not an economic or algorithmic speed claim.

Recorded passing counts: unit_tests_passed = 8. These counts describe tests, not project completion or investment effectiveness.

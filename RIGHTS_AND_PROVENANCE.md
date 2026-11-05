# Rights and provenance

Reviewed 5 November 2026 for this restricted publication package.

## Owner confirmation and scope

Subba Cecchetto explicitly confirmed: "Confermo i diritti di diffusione per il mio
codice di tesi e RO-01". The confirmation was supplied by the project owner during
preparation of this demonstration. Public inspection and reproduction of the
included demonstration are authorized within that requested publication scope.

This package does not assign an MIT, Apache, Creative Commons or other broad
open-source licence automatically. No open-source licensing badge is asserted.
The owner's confirmation records authority for this release; it is not a legal
opinion or a substitute for any later chosen formal licence.

## Preserved source evidence

The six files in `src/`, including the empty package initializer, are byte-identical
to the selected thesis sources at revision
`c19ff103d6ad6c3aa9d0baa456c45b3d19ec60bd`. Per-file SHA-256 checksums, byte counts,
last source commits and Git author names appear in `SOURCE_MANIFEST.json`.

Read-only Git history attributes the selected sources to Subba Cecchetto:

- Black-Scholes and risk metrics: first recorded in commit `e656c31` on
  31 May 2026, "Add Black-Scholes delta hedging benchmark".
- ABM history includes `6812217` (31 May), `7a9b3b7` (3 July), `8455862`
  (8 July), `99625e7` (9 July), `15b513a` (15 July) and `e97576b` (16 July 2026).
- Causal data construction: `b92a8ee` on 19 July 2026.
- Causal Black-Scholes accounting: `95dfac5` on 24 July 2026.

Git attribution supports provenance but is not by itself proof of legal title.
The selected file headers and comment scan revealed no third-party copyright
notice, external copied-source attribution or conflicting licence marker. No
notice was stripped. This is a bounded review of the allowlisted files, not a
legal certification of the entire private repository.

`run_demo.py`, `compare_outputs.py`, the portable tests and this package's
documentation were prepared for the public demonstration with AI assistance and
explicit verification. They are separate from the original scientific sources.

## Scenario parameters and data

The baseline and low-liquidity parameter values come from the existing
`get_market_regime_configurations()` function in `run_market_regimes.py`. That
source's hash is recorded in the manifest. The only scenario change is the new,
disclosed illustration seed `20261105`. No parameter optimization was performed.

Every observation in the example is generated locally by the included simulator.
The package contains no market quote, SPY cache, Bloomberg export, employer/client
file, private research matrix, credential, account, fitted neural weight or full
Git history. Previously inspected research datasets are not used as new tests.

Dependencies are installed separately from the pinned environment specification;
their source code and binary distributions are not vendored in this repository.
Dependency provenance and notices remain those of their respective distributions.

## Limits of the release

The copied modules include optional model features beyond the two scenarios run
by this wrapper. Their inclusion does not assert that all optional branches were
validated by this demonstration. The portable tests and small demonstration are
not substitutes for the private project's original full studies or protocols.


## Publication scope confirmed by the owner

On 5 November 2026 the owner authorized public publication of the verified demonstration and its use for independent technical inspection and execution. For the selected original thesis and RO-01 modules, the owner separately confirmed their rights of diffusion. This scope does not include internship materials, proprietary market data or trained weights. No general MIT, Apache or other open-source reuse licence has been selected automatically.

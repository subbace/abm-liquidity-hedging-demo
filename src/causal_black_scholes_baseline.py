"""Pure causal Black--Scholes baseline and raw-feature contract for C-02.

The module consumes immutable C-01 option windows and performs deterministic
numeric transformations only.  It has no filesystem, simulator, random-number
or model-fitting side effects.  Public numeric arrays are defensive float64
copies marked read-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.black_scholes import call_delta, european_call_price
from src.causal_hedging_data import (
    CANONICAL_FEATURE_ORDER as C01_FEATURE_ORDER,
    SCHEMA_VERSION as C01_SCHEMA_VERSION,
    CausalDataValidationError,
    CausalOptionWindow,
    ScalerStats,
    hash_causal_array,
    validate_causal_path_frame,
    validate_partition_isolation,
)
from src.risk_metrics import expected_shortfall, value_at_risk


SCHEMA_VERSION = "C02-1.0"

ABM_V042_CORE_COLUMNS: Tuple[str, ...] = (
    "time",
    "price",
    "log_return",
    "chartist_share",
    "fundamentalist_share",
    "fundamentalist_demand",
    "chartist_demand",
    "noise_demand",
    "aggregate_demand",
    "raw_liquidity",
    "liquidity",
    "stress_state",
    "stress_target",
    "noise_scale",
    "noise_multiplier",
    "p_F_to_C",
    "p_C_to_F",
)

CAUSAL_ADAPTER_COLUMNS: Tuple[str, ...] = (
    "price",
    "liquidity",
    "chartist_share",
    "aggregate_demand",
    "stress_state",
    "noise_multiplier",
)

RAW_FEATURE_ORDER: Tuple[str, ...] = (
    "log_moneyness",
    "normalized_remaining_time",
    "previous_hedge",
    "normalized_liquidity",
    "normalized_causal_rolling_volatility",
    "chartist_share",
    "normalized_aggregate_demand",
    "normalized_stress_state",
    "normalized_noise_multiplier",
)

RAW_SCALER_SOURCE_FEATURES: Tuple[str, ...] = (
    "liquidity",
    "rolling_volatility",
    "aggregate_demand",
    "stress_state",
    "noise_multiplier",
)

BASELINE_METRIC_FIELDS: Tuple[str, ...] = (
    "count",
    "mean_hedging_error",
    "standard_deviation",
    "mean_absolute_error",
    "root_mean_squared_error",
    "minimum_hedging_error",
    "maximum_hedging_error",
    "var_95",
    "expected_shortfall_95",
    "maximum_loss",
    "mean_transaction_cost",
    "median_transaction_cost",
    "mean_unit_turnover",
    "median_unit_turnover",
    "mean_notional_turnover",
    "median_notional_turnover",
)

_RESULT_ARRAY_FIELDS: Tuple[str, ...] = (
    "prices",
    "deltas",
    "trade_sizes",
    "transaction_costs",
    "cash_before_trade",
    "cash_after_trade",
    "cash_after_accrual",
)

__all__ = [
    "ABM_V042_CORE_COLUMNS",
    "BASELINE_METRIC_FIELDS",
    "CAUSAL_ADAPTER_COLUMNS",
    "RAW_FEATURE_ORDER",
    "RAW_SCALER_SOURCE_FEATURES",
    "SCHEMA_VERSION",
    "CausalBlackScholesConfig",
    "CausalBlackScholesWindowResult",
    "adapt_abm_v042_frame",
    "aggregate_path_and_regime_metrics",
    "build_raw_feature_contract",
    "build_raw_feature_row",
    "build_raw_feature_scalers",
    "canonical_baseline_result_record",
    "compute_causal_baseline_metrics",
    "evaluate_causal_black_scholes_windows",
    "simulate_causal_black_scholes_window",
    "validate_raw_feature_runtime_contract",
]


def _is_plain_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _finite_scalar(value: object, field_name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise CausalDataValidationError(
            f"{field_name} must be a finite scalar."
        ) from exc
    if not math.isfinite(converted):
        raise CausalDataValidationError(f"{field_name} must be finite.")
    return converted


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalDataValidationError(f"{field_name} must be non-empty text.")
    return value


def _readonly_float64(
    values: Sequence[float],
    *,
    field_name: str,
    expected_length: Optional[int] = None,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CausalDataValidationError(
            f"{field_name} must be convertible to float64."
        ) from exc
    if array.ndim != 1:
        raise CausalDataValidationError(f"{field_name} must be one-dimensional.")
    if expected_length is not None and len(array) != expected_length:
        raise CausalDataValidationError(
            f"{field_name} must contain exactly {expected_length} values; "
            f"observed {len(array)}."
        )
    if not np.isfinite(array).all():
        raise CausalDataValidationError(f"{field_name} contains NaN or infinity.")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def _allclose(
    observed: Sequence[float],
    expected: Sequence[float],
    *,
    field_name: str,
    atol: float,
    rtol: float,
) -> None:
    observed_array = np.asarray(observed, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    if observed_array.shape != expected_array.shape or not np.allclose(
        observed_array,
        expected_array,
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    ):
        raise CausalDataValidationError(
            f"{field_name} violates the frozen numeric identity."
        )


@dataclass(frozen=True)
class CausalBlackScholesConfig:
    """Numeric contract for the causal Black--Scholes baseline."""

    risk_free_rate: float = 0.02
    steps_per_year: int = 252
    maturity_steps: int = 63
    transaction_cost_rate: float = 0.001
    confidence_level: float = 0.95
    formula_atol: float = 1e-12
    formula_rtol: float = 1e-12
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        rate = _finite_scalar(self.risk_free_rate, "risk_free_rate")
        cost_rate = _finite_scalar(
            self.transaction_cost_rate, "transaction_cost_rate"
        )
        confidence = _finite_scalar(self.confidence_level, "confidence_level")
        atol = _finite_scalar(self.formula_atol, "formula_atol")
        rtol = _finite_scalar(self.formula_rtol, "formula_rtol")
        if not _is_plain_integer(self.steps_per_year) or self.steps_per_year != 252:
            raise CausalDataValidationError("steps_per_year is frozen at 252.")
        if not _is_plain_integer(self.maturity_steps) or self.maturity_steps != 63:
            raise CausalDataValidationError("maturity_steps is frozen at 63.")
        if cost_rate != 0.001:
            raise CausalDataValidationError(
                "transaction_cost_rate is frozen at 0.001 for C-02."
            )
        if confidence != 0.95:
            raise CausalDataValidationError("confidence_level is frozen at 0.95.")
        if atol != 1e-12 or rtol != 1e-12:
            raise CausalDataValidationError(
                "formula_atol and formula_rtol are frozen at 1e-12."
            )
        if self.schema_version != SCHEMA_VERSION:
            raise CausalDataValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}."
            )
        object.__setattr__(self, "risk_free_rate", rate)
        object.__setattr__(self, "transaction_cost_rate", cost_rate)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "formula_atol", atol)
        object.__setattr__(self, "formula_rtol", rtol)

    @property
    def growth_factor(self) -> float:
        return float(math.exp(self.risk_free_rate / self.steps_per_year))

    @property
    def maturity_years(self) -> float:
        return float(self.maturity_steps / self.steps_per_year)


@dataclass(frozen=True)
class CausalBlackScholesWindowResult:
    """Immutable result and complete 63-decision accounting trace."""

    path_id: str
    path_seed: int
    regime: str
    split: str
    window_id: str
    start_index: int
    end_index: int
    strike: float
    premium: float
    inception_volatility: float
    risk_free_rate: float
    steps_per_year: int
    transaction_cost_rate: float
    transaction_cost_convention_id: str
    prices: np.ndarray
    deltas: np.ndarray
    trade_sizes: np.ndarray
    transaction_costs: np.ndarray
    cash_before_trade: np.ndarray
    cash_after_trade: np.ndarray
    cash_after_accrual: np.ndarray
    unit_turnover: float
    notional_turnover: float
    total_transaction_cost: float
    terminal_cash: float
    terminal_stock_value: float
    terminal_portfolio: float
    payoff: float
    hedging_error: float
    loss: float
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _nonempty_text(self.path_id, "path_id")
        _nonempty_text(self.regime, "regime")
        _nonempty_text(self.split, "split")
        _nonempty_text(self.window_id, "window_id")
        _nonempty_text(
            self.transaction_cost_convention_id,
            "transaction_cost_convention_id",
        )
        if not _is_plain_integer(self.path_seed):
            raise CausalDataValidationError("path_seed must be an integer label.")
        if not _is_plain_integer(self.start_index) or not _is_plain_integer(
            self.end_index
        ):
            raise CausalDataValidationError(
                "start_index and end_index must be integers."
            )
        if self.end_index != self.start_index + 63:
            raise CausalDataValidationError(
                "Result indices violate the 63-step window contract."
            )
        if not _is_plain_integer(self.steps_per_year) or self.steps_per_year != 252:
            raise CausalDataValidationError("steps_per_year is frozen at 252.")
        if self.schema_version != SCHEMA_VERSION:
            raise CausalDataValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}."
            )

        scalar_names = (
            "strike",
            "premium",
            "inception_volatility",
            "risk_free_rate",
            "transaction_cost_rate",
            "unit_turnover",
            "notional_turnover",
            "total_transaction_cost",
            "terminal_cash",
            "terminal_stock_value",
            "terminal_portfolio",
            "payoff",
            "hedging_error",
            "loss",
        )
        scalars = {
            name: _finite_scalar(getattr(self, name), name) for name in scalar_names
        }
        if scalars["strike"] <= 0.0 or scalars["premium"] < 0.0:
            raise CausalDataValidationError(
                "strike must be positive and premium non-negative."
            )
        if scalars["inception_volatility"] < 1e-6:
            raise CausalDataValidationError(
                "inception_volatility is below the causal floor."
            )
        if scalars["transaction_cost_rate"] < 0.0:
            raise CausalDataValidationError(
                "transaction_cost_rate must be non-negative."
            )

        arrays = {
            "prices": _readonly_float64(
                self.prices, field_name="prices", expected_length=64
            )
        }
        if np.any(arrays["prices"] <= 0.0):
            raise CausalDataValidationError("prices must be strictly positive.")
        for name in _RESULT_ARRAY_FIELDS[1:]:
            arrays[name] = _readonly_float64(
                getattr(self, name), field_name=name, expected_length=63
            )
        if np.any(arrays["deltas"] < 0.0) or np.any(arrays["deltas"] > 1.0):
            raise CausalDataValidationError("deltas must remain inside [0, 1].")
        if np.any(arrays["transaction_costs"] < 0.0):
            raise CausalDataValidationError(
                "transaction_costs cannot be negative."
            )

        atol = 1e-12
        rtol = 1e-12
        expected_trades = np.diff(
            np.concatenate((np.array([0.0], dtype=np.float64), arrays["deltas"]))
        )
        _allclose(
            arrays["trade_sizes"],
            expected_trades,
            field_name="trade_sizes",
            atol=atol,
            rtol=rtol,
        )
        expected_costs = (
            scalars["transaction_cost_rate"]
            * np.abs(expected_trades)
            * arrays["prices"][:-1]
        )
        _allclose(
            arrays["transaction_costs"],
            expected_costs,
            field_name="transaction_costs",
            atol=atol,
            rtol=rtol,
        )
        if not math.isclose(
            float(arrays["cash_before_trade"][0]),
            scalars["premium"],
            abs_tol=atol,
            rel_tol=rtol,
        ):
            raise CausalDataValidationError(
                "Initial cash must equal the inception premium."
            )
        expected_cash_after_trade = (
            arrays["cash_before_trade"]
            - arrays["trade_sizes"] * arrays["prices"][:-1]
            - arrays["transaction_costs"]
        )
        _allclose(
            arrays["cash_after_trade"],
            expected_cash_after_trade,
            field_name="cash_after_trade",
            atol=atol,
            rtol=rtol,
        )
        growth_factor = math.exp(scalars["risk_free_rate"] / 252.0)
        expected_cash_after_accrual = arrays["cash_after_trade"] * growth_factor
        _allclose(
            arrays["cash_after_accrual"],
            expected_cash_after_accrual,
            field_name="cash_after_accrual",
            atol=atol,
            rtol=rtol,
        )
        _allclose(
            arrays["cash_before_trade"][1:],
            arrays["cash_after_accrual"][:-1],
            field_name="cash recurrence",
            atol=atol,
            rtol=rtol,
        )

        expected_unit_turnover = float(np.sum(np.abs(expected_trades)))
        expected_notional_turnover = float(
            np.sum(np.abs(expected_trades) * arrays["prices"][:-1])
        )
        expected_total_cost = float(np.sum(expected_costs))
        scalar_identities = {
            "unit_turnover": expected_unit_turnover,
            "notional_turnover": expected_notional_turnover,
            "total_transaction_cost": expected_total_cost,
            "terminal_cash": float(arrays["cash_after_accrual"][-1]),
            "terminal_stock_value": float(
                arrays["deltas"][-1] * arrays["prices"][-1]
            ),
        }
        scalar_identities["terminal_portfolio"] = (
            scalar_identities["terminal_cash"]
            + scalar_identities["terminal_stock_value"]
        )
        scalar_identities["payoff"] = float(
            max(arrays["prices"][-1] - scalars["strike"], 0.0)
        )
        scalar_identities["hedging_error"] = (
            scalar_identities["terminal_portfolio"]
            - scalar_identities["payoff"]
        )
        scalar_identities["loss"] = -scalar_identities["hedging_error"]
        for name, expected in scalar_identities.items():
            if not math.isclose(
                scalars[name], expected, abs_tol=atol, rel_tol=rtol
            ):
                raise CausalDataValidationError(
                    f"{name} violates the frozen accounting identity."
                )

        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)

    @property
    def hedge_positions(self) -> np.ndarray:
        """Compatibility alias for the 63 Black--Scholes deltas."""

        return self.deltas

    @property
    def trades(self) -> np.ndarray:
        """Compatibility alias for the 63 signed trades."""

        return self.trade_sizes

    @property
    def costs(self) -> np.ndarray:
        """Compatibility alias for the 63 transaction costs."""

        return self.transaction_costs

    @property
    def final_portfolio_value(self) -> float:
        """Compatibility alias for the terminal portfolio before payoff."""

        return self.terminal_portfolio


def _resolved_config(
    config: Optional[CausalBlackScholesConfig],
) -> CausalBlackScholesConfig:
    if config is None:
        return CausalBlackScholesConfig()
    if not isinstance(config, CausalBlackScholesConfig):
        raise CausalDataValidationError(
            "config must be a CausalBlackScholesConfig instance."
        )
    return config


def adapt_abm_v042_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate an ABM v0.4.2 core and project the six C-01 state columns.

    Additional columns are ignored only after the complete 17-column core has
    been found and validated.  The producer's first return is intentionally not
    forwarded: C-01 reconstructs the required NaN sentinel from prices.
    """

    if not isinstance(frame, pd.DataFrame):
        raise CausalDataValidationError("frame must be a pandas DataFrame.")
    if len(frame) != 5001:
        raise CausalDataValidationError(
            "An official ABM v0.4.2 path must contain exactly 5,001 rows."
        )
    if isinstance(frame.index, pd.MultiIndex) or not frame.index.is_unique:
        raise CausalDataValidationError(
            "ABM frame index must be unique and one-dimensional."
        )
    expected_index = np.arange(len(frame), dtype=np.int64)
    try:
        observed_index = frame.index.to_numpy(dtype=np.int64, copy=True)
    except (TypeError, ValueError) as exc:
        raise CausalDataValidationError(
            "ABM frame index must be integer and zero-based."
        ) from exc
    if not np.array_equal(observed_index, expected_index):
        raise CausalDataValidationError(
            "ABM frame index must be contiguous and zero-based."
        )
    if not frame.columns.is_unique:
        raise CausalDataValidationError("ABM frame columns must be unique.")
    missing = [name for name in ABM_V042_CORE_COLUMNS if name not in frame.columns]
    if missing:
        raise CausalDataValidationError(
            "Missing ABM v0.4.2 columns: " + ", ".join(missing)
        )

    core: Dict[str, np.ndarray] = {}
    for column in ABM_V042_CORE_COLUMNS:
        try:
            values = frame[column].to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise CausalDataValidationError(
                f"ABM core column {column} must be numeric."
            ) from exc
        if values.ndim != 1 or len(values) != 5001 or not np.isfinite(values).all():
            raise CausalDataValidationError(
                f"ABM core column {column} must contain 5,001 finite values."
            )
        core[column] = values

    if not np.array_equal(core["time"], expected_index.astype(np.float64)):
        raise CausalDataValidationError("ABM time must equal the row index.")
    if np.any(core["price"] <= 0.0):
        raise CausalDataValidationError("ABM prices must be strictly positive.")
    if np.any(core["raw_liquidity"] <= 0.0) or np.any(core["liquidity"] <= 0.0):
        raise CausalDataValidationError("ABM liquidity must be strictly positive.")
    if np.any(core["noise_scale"] <= 0.0) or np.any(
        core["noise_multiplier"] <= 0.0
    ):
        raise CausalDataValidationError(
            "ABM noise scale and multiplier must be strictly positive."
        )
    for probability_name in (
        "chartist_share",
        "fundamentalist_share",
        "p_F_to_C",
        "p_C_to_F",
    ):
        if np.any(core[probability_name] < 0.0) or np.any(
            core[probability_name] > 1.0
        ):
            raise CausalDataValidationError(
                f"{probability_name} must remain inside [0, 1]."
            )
    if not np.allclose(
        core["chartist_share"] + core["fundamentalist_share"],
        np.ones(5001, dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise CausalDataValidationError(
            "Agent shares must sum to one at every path time."
        )
    if not math.isclose(float(core["log_return"][0]), 0.0, abs_tol=1e-12):
        raise CausalDataValidationError(
            "The v0.4.2 producer return at row zero must be 0.0."
        )
    expected_returns = np.log(core["price"][1:] / core["price"][:-1])
    if not np.allclose(
        core["log_return"][1:],
        expected_returns,
        atol=1e-12,
        rtol=1e-12,
        equal_nan=False,
    ):
        raise CausalDataValidationError(
            "ABM producer returns do not match the price path."
        )

    projected = pd.DataFrame(
        {
            name: np.array(core[name], dtype=np.float64, copy=True)
            for name in CAUSAL_ADAPTER_COLUMNS
        },
        index=pd.RangeIndex(start=0, stop=5001, step=1),
    )
    validated = validate_causal_path_frame(projected)
    if not np.isnan(float(validated.loc[0, "log_return"])):
        raise CausalDataValidationError(
            "C-01 did not derive the required first-return NaN sentinel."
        )
    return projected.copy(deep=True)


def simulate_causal_black_scholes_window(
    window: CausalOptionWindow,
    config: Optional[CausalBlackScholesConfig] = None,
) -> CausalBlackScholesWindowResult:
    """Evaluate one immutable C-01 window under the frozen C-02 accounting."""

    if not isinstance(window, CausalOptionWindow):
        raise CausalDataValidationError(
            "window must be a CausalOptionWindow instance."
        )
    resolved = _resolved_config(config)
    if not math.isclose(
        float(window.risk_free_rate),
        resolved.risk_free_rate,
        abs_tol=resolved.formula_atol,
        rel_tol=resolved.formula_rtol,
    ):
        raise CausalDataValidationError(
            "Window and baseline risk-free rates must be identical."
        )
    if not math.isclose(
        float(window.maturity_years),
        resolved.maturity_years,
        abs_tol=resolved.formula_atol,
        rel_tol=resolved.formula_rtol,
    ):
        raise CausalDataValidationError(
            "Window maturity violates the 63/252 contract."
        )

    prices = _readonly_float64(
        window.prices, field_name="window.prices", expected_length=64
    )
    strike = float(window.strike)
    sigma = float(window.inception_volatility)
    premium = float(
        european_call_price(
            S=float(prices[0]),
            K=strike,
            r=resolved.risk_free_rate,
            sigma=sigma,
            tau=resolved.maturity_years,
        )
    )
    if not math.isfinite(premium) or premium < 0.0:
        raise CausalDataValidationError("Black--Scholes premium is invalid.")
    if not math.isclose(
        premium,
        float(window.premium),
        abs_tol=resolved.formula_atol,
        rel_tol=resolved.formula_rtol,
    ):
        raise CausalDataValidationError(
            "Window premium does not use the fixed causal inception sigma."
        )

    deltas = np.empty(63, dtype=np.float64)
    for decision_index in range(63):
        tau = float((63 - decision_index) / 252)
        delta = float(
            call_delta(
                S=float(prices[decision_index]),
                K=strike,
                r=resolved.risk_free_rate,
                sigma=sigma,
                tau=tau,
            )
        )
        if not math.isfinite(delta) or delta < 0.0 or delta > 1.0:
            raise CausalDataValidationError(
                f"Black--Scholes delta is invalid at decision {decision_index}."
            )
        deltas[decision_index] = delta
    _allclose(
        deltas,
        window.black_scholes_delta_path,
        field_name="fixed-inception-sigma delta path",
        atol=resolved.formula_atol,
        rtol=resolved.formula_rtol,
    )

    previous_positions = np.concatenate(
        (np.array([0.0], dtype=np.float64), deltas[:-1])
    )
    trades = deltas - previous_positions
    costs = (
        resolved.transaction_cost_rate * np.abs(trades) * prices[:-1]
    ).astype(np.float64, copy=False)
    cash_before = np.empty(63, dtype=np.float64)
    cash_after_trade = np.empty(63, dtype=np.float64)
    cash_after_accrual = np.empty(63, dtype=np.float64)
    cash_before[0] = premium
    growth_factor = resolved.growth_factor
    for decision_index in range(63):
        if decision_index > 0:
            cash_before[decision_index] = cash_after_accrual[decision_index - 1]
        cash_after_trade[decision_index] = (
            cash_before[decision_index]
            - trades[decision_index] * prices[decision_index]
            - costs[decision_index]
        )
        cash_after_accrual[decision_index] = (
            cash_after_trade[decision_index] * growth_factor
        )
    trace_arrays = (
        deltas,
        trades,
        costs,
        cash_before,
        cash_after_trade,
        cash_after_accrual,
    )
    if not all(np.isfinite(array).all() for array in trace_arrays):
        raise CausalDataValidationError(
            "The Black--Scholes accounting trace contains a non-finite value."
        )

    unit_turnover = float(np.sum(np.abs(trades), dtype=np.float64))
    notional_turnover = float(
        np.sum(np.abs(trades) * prices[:-1], dtype=np.float64)
    )
    total_transaction_cost = float(np.sum(costs, dtype=np.float64))
    terminal_cash = float(cash_after_accrual[-1])
    terminal_stock_value = float(deltas[-1] * prices[-1])
    terminal_portfolio = float(terminal_cash + terminal_stock_value)
    payoff = float(max(prices[-1] - strike, 0.0))
    hedging_error = float(terminal_portfolio - payoff)
    loss = float(-hedging_error)

    return CausalBlackScholesWindowResult(
        path_id=window.path_id,
        path_seed=int(window.path_seed),
        regime=window.regime,
        split=window.split,
        window_id=window.window_id,
        start_index=int(window.start_index),
        end_index=int(window.end_index),
        strike=strike,
        premium=premium,
        inception_volatility=sigma,
        risk_free_rate=resolved.risk_free_rate,
        steps_per_year=resolved.steps_per_year,
        transaction_cost_rate=resolved.transaction_cost_rate,
        transaction_cost_convention_id=window.transaction_cost_convention_id,
        prices=prices,
        deltas=deltas,
        trade_sizes=trades,
        transaction_costs=costs,
        cash_before_trade=cash_before,
        cash_after_trade=cash_after_trade,
        cash_after_accrual=cash_after_accrual,
        unit_turnover=unit_turnover,
        notional_turnover=notional_turnover,
        total_transaction_cost=total_transaction_cost,
        terminal_cash=terminal_cash,
        terminal_stock_value=terminal_stock_value,
        terminal_portfolio=terminal_portfolio,
        payoff=payoff,
        hedging_error=hedging_error,
        loss=loss,
        schema_version=resolved.schema_version,
    )


def evaluate_causal_black_scholes_windows(
    windows: Sequence[CausalOptionWindow],
    config: Optional[CausalBlackScholesConfig] = None,
) -> Tuple[CausalBlackScholesWindowResult, ...]:
    """Evaluate a unique collection in stable ``window_id`` order."""

    if not windows:
        raise CausalDataValidationError("windows cannot be empty.")
    if any(not isinstance(window, CausalOptionWindow) for window in windows):
        raise CausalDataValidationError(
            "windows must contain only CausalOptionWindow objects."
        )
    ordered = tuple(sorted(windows, key=lambda item: item.window_id))
    window_ids = tuple(window.window_id for window in ordered)
    if len(window_ids) != len(set(window_ids)):
        raise CausalDataValidationError("window_id values must be unique.")
    validate_partition_isolation(ordered)
    return tuple(
        simulate_causal_black_scholes_window(window, config=config)
        for window in ordered
    )


def _validated_results(
    results: Sequence[CausalBlackScholesWindowResult],
    *,
    minimum_count: int,
) -> Tuple[CausalBlackScholesWindowResult, ...]:
    if not results:
        raise CausalDataValidationError("results cannot be empty.")
    if any(
        not isinstance(result, CausalBlackScholesWindowResult)
        for result in results
    ):
        raise CausalDataValidationError(
            "results must contain only CausalBlackScholesWindowResult objects."
        )
    ordered = tuple(sorted(results, key=lambda item: item.window_id))
    identifiers = tuple(result.window_id for result in ordered)
    if len(identifiers) != len(set(identifiers)):
        raise CausalDataValidationError("Result window_id values must be unique.")
    if len(ordered) < minimum_count:
        raise CausalDataValidationError(
            f"At least {minimum_count} results are required."
        )
    return ordered


def compute_causal_baseline_metrics(
    results: Sequence[CausalBlackScholesWindowResult],
) -> Dict[str, object]:
    """Compute the frozen finite descriptive metrics for one result group."""

    ordered = _validated_results(results, minimum_count=2)
    errors = np.asarray(
        [result.hedging_error for result in ordered], dtype=np.float64
    )
    losses = np.asarray([result.loss for result in ordered], dtype=np.float64)
    costs = np.asarray(
        [result.total_transaction_cost for result in ordered], dtype=np.float64
    )
    unit_turnovers = np.asarray(
        [result.unit_turnover for result in ordered], dtype=np.float64
    )
    notional_turnovers = np.asarray(
        [result.notional_turnover for result in ordered], dtype=np.float64
    )
    for name, values in (
        ("hedging errors", errors),
        ("losses", losses),
        ("transaction costs", costs),
        ("unit turnovers", unit_turnovers),
        ("notional turnovers", notional_turnovers),
    ):
        if values.ndim != 1 or not np.isfinite(values).all():
            raise CausalDataValidationError(f"{name} must be finite.")
    if not np.array_equal(losses, -errors):
        raise CausalDataValidationError("loss must equal negative hedging error.")

    var95 = float(value_at_risk(losses=losses, confidence_level=0.95))
    es95 = float(expected_shortfall(losses=losses, confidence_level=0.95))
    metrics: Dict[str, object] = {
        "count": int(len(ordered)),
        "mean_hedging_error": float(np.mean(errors, dtype=np.float64)),
        "standard_deviation": float(np.std(errors, ddof=1, dtype=np.float64)),
        "mean_absolute_error": float(np.mean(np.abs(errors), dtype=np.float64)),
        "root_mean_squared_error": float(
            np.sqrt(np.mean(np.square(errors), dtype=np.float64))
        ),
        "minimum_hedging_error": float(np.min(errors)),
        "maximum_hedging_error": float(np.max(errors)),
        "var_95": var95,
        "expected_shortfall_95": es95,
        "maximum_loss": float(np.max(losses)),
        "mean_transaction_cost": float(np.mean(costs, dtype=np.float64)),
        "median_transaction_cost": float(np.median(costs)),
        "mean_unit_turnover": float(
            np.mean(unit_turnovers, dtype=np.float64)
        ),
        "median_unit_turnover": float(np.median(unit_turnovers)),
        "mean_notional_turnover": float(
            np.mean(notional_turnovers, dtype=np.float64)
        ),
        "median_notional_turnover": float(np.median(notional_turnovers)),
    }
    if tuple(metrics) != BASELINE_METRIC_FIELDS:
        raise CausalDataValidationError("Internal metric order is inconsistent.")
    numeric_values = [
        float(value) for key, value in metrics.items() if key != "count"
    ]
    if not np.isfinite(np.asarray(numeric_values, dtype=np.float64)).all():
        raise CausalDataValidationError("A baseline metric is non-finite.")
    return metrics


def aggregate_path_and_regime_metrics(
    results: Sequence[CausalBlackScholesWindowResult],
) -> Dict[str, Tuple[Dict[str, object], ...]]:
    """Build primary path metrics and equal-path-weight regime summaries."""

    ordered = _validated_results(results, minimum_count=2)
    paths: Dict[
        Tuple[str, int, str, str], list[CausalBlackScholesWindowResult]
    ] = {}
    path_identity_by_id: Dict[str, Tuple[int, str, str]] = {}
    for result in ordered:
        identity = (int(result.path_seed), result.regime, result.split)
        prior = path_identity_by_id.setdefault(result.path_id, identity)
        if prior != identity:
            raise CausalDataValidationError(
                f"Inconsistent metadata for path_id {result.path_id!r}."
            )
        key = (result.path_id, *identity)
        paths.setdefault(key, []).append(result)

    path_rows = []
    for key in sorted(paths):
        path_id, path_seed, regime, split = key
        metrics = compute_causal_baseline_metrics(paths[key])
        row: Dict[str, object] = {
            "path_id": path_id,
            "path_seed": int(path_seed),
            "regime": regime,
            "split": split,
        }
        row.update(metrics)
        path_rows.append(row)

    regimes: Dict[Tuple[str, str], list[Dict[str, object]]] = {}
    for row in path_rows:
        key = (str(row["split"]), str(row["regime"]))
        regimes.setdefault(key, []).append(row)
    regime_rows = []
    non_count_metrics = BASELINE_METRIC_FIELDS[1:]
    for key in sorted(regimes):
        split, regime = key
        contributing_paths = sorted(
            regimes[key], key=lambda row: str(row["path_id"])
        )
        regime_row: Dict[str, object] = {
            "split": split,
            "regime": regime,
            "path_count": int(len(contributing_paths)),
            "aggregation_basis": "equal_weight_path_metrics",
            "count": int(
                sum(int(row["count"]) for row in contributing_paths)
            ),
        }
        for metric_name in non_count_metrics:
            values = np.asarray(
                [float(row[metric_name]) for row in contributing_paths],
                dtype=np.float64,
            )
            if not np.isfinite(values).all():
                raise CausalDataValidationError(
                    f"Path metric {metric_name} is non-finite."
                )
            regime_row[metric_name] = float(
                np.mean(values, dtype=np.float64)
            )
        regime_rows.append(regime_row)
    return {
        "path_summaries": tuple(path_rows),
        "regime_summaries": tuple(regime_rows),
    }


def build_raw_feature_contract() -> Dict[str, object]:
    """Return the fixed, JSON-compatible nine-feature runtime contract."""

    definitions = (
        {
            "feature_name": "log_moneyness",
            "source": "log(S_j / K)",
            "standardized": False,
            "runtime_only": True,
        },
        {
            "feature_name": "normalized_remaining_time",
            "source": "(63 - j) / 63",
            "standardized": False,
            "runtime_only": True,
        },
        {
            "feature_name": "previous_hedge",
            "source": "policy-owned prior hedge",
            "standardized": False,
            "runtime_only": True,
        },
        {
            "feature_name": "normalized_liquidity",
            "source": "liquidity",
            "standardized": True,
            "runtime_only": False,
        },
        {
            "feature_name": "normalized_causal_rolling_volatility",
            "source": "rolling_volatility",
            "standardized": True,
            "runtime_only": False,
        },
        {
            "feature_name": "chartist_share",
            "source": "chartist_share",
            "standardized": False,
            "runtime_only": False,
        },
        {
            "feature_name": "normalized_aggregate_demand",
            "source": "aggregate_demand",
            "standardized": True,
            "runtime_only": False,
        },
        {
            "feature_name": "normalized_stress_state",
            "source": "stress_state",
            "standardized": True,
            "runtime_only": False,
        },
        {
            "feature_name": "normalized_noise_multiplier",
            "source": "noise_multiplier",
            "standardized": True,
            "runtime_only": False,
        },
    )
    if tuple(item["feature_name"] for item in definitions) != RAW_FEATURE_ORDER:
        raise CausalDataValidationError(
            "Internal raw-feature order is inconsistent."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "c01_schema_version": C01_SCHEMA_VERSION,
        "feature_count": 9,
        "feature_order": RAW_FEATURE_ORDER,
        "scaler_source_features": RAW_SCALER_SOURCE_FEATURES,
        "runtime_only_features": (
            "log_moneyness",
            "normalized_remaining_time",
            "previous_hedge",
        ),
        "unscaled_state_features": ("chartist_share",),
        "previous_hedge_static_storage_allowed": False,
        "terminal_row_allowed": False,
        "clipping_allowed": False,
        "definitions": definitions,
    }


def build_raw_feature_scalers(
    windows: Sequence[CausalOptionWindow],
) -> Mapping[str, ScalerStats]:
    """Fit five population scalers on window-expanded train decisions only."""

    if not windows:
        raise CausalDataValidationError("Scaler windows cannot be empty.")
    if any(not isinstance(window, CausalOptionWindow) for window in windows):
        raise CausalDataValidationError(
            "Scaler windows must be CausalOptionWindow objects."
        )
    validate_partition_isolation(tuple(windows))
    training_windows = tuple(
        sorted(
            (window for window in windows if window.split == "train"),
            key=lambda item: item.window_id,
        )
    )
    if not training_windows:
        raise CausalDataValidationError(
            "At least one training window is required for scaler fitting."
        )
    window_ids = tuple(window.window_id for window in training_windows)
    if len(window_ids) != len(set(window_ids)):
        raise CausalDataValidationError(
            "Training window identifiers must be unique."
        )
    observation_count = 63 * len(training_windows)
    scalers: Dict[str, ScalerStats] = {}
    for source_name in RAW_SCALER_SOURCE_FEATURES:
        values = np.concatenate(
            [
                np.asarray(getattr(window, source_name), dtype=np.float64)
                for window in training_windows
            ]
        ).astype(np.float64, copy=False)
        if len(values) != observation_count or not np.isfinite(values).all():
            raise CausalDataValidationError(
                f"Scaler source {source_name} has invalid decision rows."
            )
        mean = float(np.mean(values, dtype=np.float64))
        standard_deviation = float(np.std(values, ddof=0, dtype=np.float64))
        if not math.isfinite(mean) or not math.isfinite(standard_deviation):
            raise CausalDataValidationError(
                f"Scaler statistics for {source_name} are non-finite."
            )
        if standard_deviation <= 0.0:
            raise CausalDataValidationError(
                f"Scaler standard deviation for {source_name} is non-positive."
            )
        scalers[source_name] = ScalerStats(
            feature_name=source_name,
            mean=mean,
            standard_deviation=standard_deviation,
            observation_count=observation_count,
            ddof=0,
            training_window_ids=window_ids,
            schema_version=C01_SCHEMA_VERSION,
        )
    return MappingProxyType(scalers)


def _validated_raw_scalers(
    scalers: Mapping[str, ScalerStats],
) -> Mapping[str, ScalerStats]:
    if not isinstance(scalers, Mapping):
        raise CausalDataValidationError("scalers must be a mapping.")
    if tuple(scalers.keys()) != RAW_SCALER_SOURCE_FEATURES:
        raise CausalDataValidationError(
            "Scaler keys and order must equal the five frozen source features."
        )
    training_ids: Optional[Tuple[str, ...]] = None
    observation_count: Optional[int] = None
    checked: Dict[str, ScalerStats] = {}
    for source_name in RAW_SCALER_SOURCE_FEATURES:
        scaler = scalers[source_name]
        if not isinstance(scaler, ScalerStats):
            raise CausalDataValidationError(
                f"Scaler for {source_name} is foreign to the C-01 contract."
            )
        if scaler.feature_name != source_name:
            raise CausalDataValidationError(
                f"Scaler identity mismatch for {source_name}."
            )
        if scaler.schema_version != C01_SCHEMA_VERSION:
            raise CausalDataValidationError(
                f"Scaler schema mismatch for {source_name}."
            )
        if not math.isfinite(float(scaler.mean)) or not math.isfinite(
            float(scaler.standard_deviation)
        ):
            raise CausalDataValidationError(
                f"Scaler for {source_name} is non-finite."
            )
        if float(scaler.standard_deviation) <= 0.0:
            raise CausalDataValidationError(
                f"Scaler for {source_name} has non-positive deviation."
            )
        if training_ids is None:
            training_ids = tuple(scaler.training_window_ids)
            observation_count = int(scaler.observation_count)
        elif tuple(scaler.training_window_ids) != training_ids or int(
            scaler.observation_count
        ) != observation_count:
            raise CausalDataValidationError(
                "All raw-feature scalers must share one training population."
            )
        checked[source_name] = scaler
    assert training_ids is not None
    assert observation_count is not None
    if observation_count != 63 * len(training_ids):
        raise CausalDataValidationError(
            "Scaler observations must equal 63 times the training-window count."
        )
    return MappingProxyType(checked)


def _raw_feature_row(
    window: CausalOptionWindow,
    decision_index: int,
    previous_hedge: float,
    scalers: Mapping[str, ScalerStats],
) -> np.ndarray:
    if not isinstance(window, CausalOptionWindow):
        raise CausalDataValidationError(
            "window must be a CausalOptionWindow instance."
        )
    if not _is_plain_integer(decision_index):
        raise CausalDataValidationError("decision_index must be an integer.")
    index = int(decision_index)
    if index < 0 or index >= 63:
        raise CausalDataValidationError(
            "decision_index must be in [0, 62]; terminal row 63 is excluded."
        )
    previous = _finite_scalar(previous_hedge, "previous_hedge")
    checked_scalers = _validated_raw_scalers(scalers)
    chartist_share = float(window.chartist_share[index])
    if not math.isfinite(chartist_share) or not 0.0 <= chartist_share <= 1.0:
        raise CausalDataValidationError(
            "chartist_share must be finite and inside [0, 1]."
        )

    def standardized(source_name: str, raw_value: float) -> float:
        value = _finite_scalar(raw_value, source_name)
        scaler = checked_scalers[source_name]
        normalized = (value - float(scaler.mean)) / float(
            scaler.standard_deviation
        )
        if not math.isfinite(normalized):
            raise CausalDataValidationError(
                f"Normalized feature for {source_name} is non-finite."
            )
        return float(normalized)

    values = np.asarray(
        (
            math.log(float(window.prices[index]) / float(window.strike)),
            float((63 - index) / 63),
            previous,
            standardized("liquidity", float(window.liquidity[index])),
            standardized(
                "rolling_volatility", float(window.rolling_volatility[index])
            ),
            chartist_share,
            standardized(
                "aggregate_demand", float(window.aggregate_demand[index])
            ),
            standardized("stress_state", float(window.stress_state[index])),
            standardized(
                "noise_multiplier", float(window.noise_multiplier[index])
            ),
        ),
        dtype=np.float64,
    )
    if values.shape != (9,) or not np.isfinite(values).all():
        raise CausalDataValidationError("Raw feature row is invalid.")
    values.setflags(write=False)
    return values


def build_raw_feature_row(
    window: CausalOptionWindow,
    decision_index: int,
    previous_hedge: float,
    scalers: Mapping[str, ScalerStats],
) -> np.ndarray:
    """Build one unclipped nine-feature row with explicit policy state."""

    return _raw_feature_row(window, decision_index, previous_hedge, scalers)


def validate_raw_feature_runtime_contract(
    window: CausalOptionWindow,
    decision_index: int,
    previous_hedge: float,
    scalers: Mapping[str, ScalerStats],
) -> Mapping[str, object]:
    """Fail closed and return an immutable audit for one runtime feature row."""

    feature_vector = _raw_feature_row(
        window, decision_index, previous_hedge, scalers
    )
    if hasattr(window, "previous_hedge"):
        raise CausalDataValidationError(
            "previous_hedge must not be stored in the static C-01 window."
        )
    audit: Dict[str, object] = {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "window_id": window.window_id,
        "decision_index": int(decision_index),
        "feature_order": RAW_FEATURE_ORDER,
        "feature_count": 9,
        "previous_hedge_runtime_only": True,
        "previous_hedge_static_storage_present": False,
        "terminal_row_excluded": True,
        "chartist_share_unscaled": True,
        "standardized_feature_count": 5,
        "clipping_applied": False,
        "feature_vector": feature_vector,
    }
    return MappingProxyType(audit)


def canonical_baseline_result_record(
    result: CausalBlackScholesWindowResult,
) -> Dict[str, object]:
    """Return a deterministic JSON-compatible core record for one window."""

    if not isinstance(result, CausalBlackScholesWindowResult):
        raise CausalDataValidationError(
            "result must be a CausalBlackScholesWindowResult instance."
        )
    array_hashes = {
        name: hash_causal_array(name, getattr(result, name))
        for name in _RESULT_ARRAY_FIELDS
    }
    return {
        "schema_version": result.schema_version,
        "path_id": result.path_id,
        "path_seed": int(result.path_seed),
        "regime": result.regime,
        "split": result.split,
        "window_id": result.window_id,
        "start_index": int(result.start_index),
        "end_index": int(result.end_index),
        "strike": float(result.strike),
        "premium": float(result.premium),
        "inception_volatility": float(result.inception_volatility),
        "risk_free_rate": float(result.risk_free_rate),
        "steps_per_year": int(result.steps_per_year),
        "transaction_cost_rate": float(result.transaction_cost_rate),
        "transaction_cost_convention_id": (
            result.transaction_cost_convention_id
        ),
        "trade_count": 63,
        "subsequent_rebalance_count": 62,
        "terminal_rebalance_count": 0,
        "unit_turnover": float(result.unit_turnover),
        "notional_turnover": float(result.notional_turnover),
        "total_transaction_cost": float(result.total_transaction_cost),
        "terminal_cash": float(result.terminal_cash),
        "terminal_stock_value": float(result.terminal_stock_value),
        "terminal_portfolio": float(result.terminal_portfolio),
        "payoff": float(result.payoff),
        "hedging_error": float(result.hedging_error),
        "loss": float(result.loss),
        "array_hashes": array_hashes,
    }

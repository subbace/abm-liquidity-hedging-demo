"""Causal hedging-data construction proposed for checkpoint C-01.

This staging module is intentionally autonomous from the legacy hedging data
pipeline.  It performs deterministic, analytic data preparation only: no
filesystem I/O, random generation, training, strategy evaluation, risk metric,
or mutable global state is present.

The only imported project mathematics are the pure Black--Scholes call price
and call delta functions.  All computational arrays use float64.  Arrays held
by public frozen records, and arrays returned by public numeric functions, are
defensive copies marked read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.black_scholes import call_delta, european_call_price


SCHEMA_VERSION = "1.0"
ALLOWED_SPLITS: Tuple[str, ...] = ("train", "validation", "test")
REQUIRED_PATH_COLUMNS: Tuple[str, ...] = (
    "price",
    "liquidity",
    "chartist_share",
    "aggregate_demand",
    "stress_state",
    "noise_multiplier",
)
CANONICAL_FEATURE_ORDER: Tuple[str, ...] = (
    "rolling_volatility",
    "chartist_share",
    "aggregate_demand",
    "stress_state",
    "noise_multiplier",
    "liquidity",
)

__all__ = [
    "ALLOWED_SPLITS",
    "CANONICAL_FEATURE_ORDER",
    "REQUIRED_PATH_COLUMNS",
    "SCHEMA_VERSION",
    "CausalDataValidationError",
    "CausalWindowConfig",
    "CausalPathPartition",
    "ScalerStats",
    "CausalOptionWindow",
    "validate_causal_path_frame",
    "compute_causal_rolling_volatility",
    "estimate_inception_volatility",
    "build_causal_option_windows",
    "assign_path_partitions",
    "validate_partition_isolation",
    "fit_training_only_scalers",
    "build_causal_black_scholes_delta_path",
    "hash_causal_array",
    "canonical_window_record",
]


class CausalDataValidationError(ValueError):
    """Fail-closed validation error, optionally carrying an audit report."""

    def __init__(
        self,
        message: str,
        *,
        report: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__(message)
        self.report = dict(report) if report is not None else None


def _is_plain_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _require_nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalDataValidationError(f"{field_name} must be non-empty text.")
    return value


def _require_finite_scalar(value: object, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CausalDataValidationError(
            f"{field_name} must be a finite scalar."
        ) from exc
    if not math.isfinite(result):
        raise CausalDataValidationError(f"{field_name} must be finite.")
    return result


def _readonly_float64_array(
    values: Sequence[float],
    *,
    field_name: str,
    expected_length: Optional[int] = None,
    allow_nan: bool = False,
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
    if allow_nan:
        if np.isinf(array).any():
            raise CausalDataValidationError(f"{field_name} contains infinity.")
    elif not np.isfinite(array).all():
        raise CausalDataValidationError(f"{field_name} contains NaN or infinity.")
    copied = np.array(array, dtype=np.float64, order="C", copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class CausalWindowConfig:
    """Frozen canonical C-01 temporal and numeric constants."""

    maturity_steps: int = 63
    stride: int = 20
    earliest_start_index: int = 63
    rolling_lookback: int = 20
    inception_lookback: int = 63
    steps_per_year: int = 252
    volatility_floor: float = 1e-6
    formula_atol: float = 1e-12
    formula_rtol: float = 1e-12
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        integer_contract = {
            "maturity_steps": (self.maturity_steps, 63),
            "stride": (self.stride, 20),
            "earliest_start_index": (self.earliest_start_index, 63),
            "rolling_lookback": (self.rolling_lookback, 20),
            "inception_lookback": (self.inception_lookback, 63),
            "steps_per_year": (self.steps_per_year, 252),
        }
        for field_name, (observed, expected) in integer_contract.items():
            if not _is_plain_integer(observed) or int(observed) != expected:
                raise CausalDataValidationError(
                    f"{field_name} is frozen at {expected}; observed {observed!r}."
                )
        float_contract = {
            "volatility_floor": (self.volatility_floor, 1e-6),
            "formula_atol": (self.formula_atol, 1e-12),
            "formula_rtol": (self.formula_rtol, 1e-12),
        }
        for field_name, (observed, expected) in float_contract.items():
            value = _require_finite_scalar(observed, field_name)
            if value != expected:
                raise CausalDataValidationError(
                    f"{field_name} is frozen at {expected}; observed {value}."
                )
        if self.schema_version != SCHEMA_VERSION:
            raise CausalDataValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}."
            )

    @property
    def maturity_years(self) -> float:
        """Canonical maturity in years."""

        return float(self.maturity_steps / self.steps_per_year)

    def canonical_start_indices(self, number_of_prices: int) -> Tuple[int, ...]:
        """Return the deterministic admissible start sequence."""

        if not _is_plain_integer(number_of_prices) or int(number_of_prices) <= 0:
            raise CausalDataValidationError(
                "number_of_prices must be a positive integer."
            )
        last_start = int(number_of_prices) - self.maturity_steps - 1
        if last_start < self.earliest_start_index:
            return tuple()
        return tuple(
            range(
                self.earliest_start_index,
                last_start + 1,
                self.stride,
            )
        )


@dataclass(frozen=True, order=True)
class CausalPathPartition:
    """Explicit deterministic path-level split assignment."""

    path_id: str
    path_seed: int
    regime: str
    split: str

    def __post_init__(self) -> None:
        _require_nonempty_text(self.path_id, "path_id")
        _require_nonempty_text(self.regime, "regime")
        if not _is_plain_integer(self.path_seed):
            raise CausalDataValidationError("path_seed must be an integer label.")
        if self.split not in ALLOWED_SPLITS:
            raise CausalDataValidationError(
                f"split must be one of {ALLOWED_SPLITS}; observed {self.split!r}."
            )


@dataclass(frozen=True)
class ScalerStats:
    """Immutable population scaler statistics for one canonical feature."""

    feature_name: str
    mean: float
    standard_deviation: float
    observation_count: int
    ddof: int
    training_window_ids: Tuple[str, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_text(self.feature_name, "feature_name")
        if self.feature_name not in CANONICAL_FEATURE_ORDER:
            raise CausalDataValidationError(
                f"Unknown scaler feature: {self.feature_name!r}."
            )
        _require_finite_scalar(self.mean, "mean")
        standard_deviation = _require_finite_scalar(
            self.standard_deviation, "standard_deviation"
        )
        if standard_deviation <= 0.0:
            raise CausalDataValidationError(
                "standard_deviation must be strictly positive."
            )
        if not _is_plain_integer(self.observation_count) or self.observation_count <= 0:
            raise CausalDataValidationError(
                "observation_count must be a positive integer."
            )
        if self.ddof != 0:
            raise CausalDataValidationError("Scaler ddof is frozen at 0.")
        if not self.training_window_ids:
            raise CausalDataValidationError("training_window_ids cannot be empty.")
        if tuple(sorted(self.training_window_ids)) != self.training_window_ids:
            raise CausalDataValidationError(
                "training_window_ids must be in stable sorted order."
            )
        if len(set(self.training_window_ids)) != len(self.training_window_ids):
            raise CausalDataValidationError(
                "training_window_ids must contain no duplicates."
            )
        if self.schema_version != SCHEMA_VERSION:
            raise CausalDataValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}."
            )

    @property
    def std(self) -> float:
        """Compatibility alias for the frozen population standard deviation."""

        return self.standard_deviation

    @property
    def count(self) -> int:
        """Compatibility alias for the number of training decision rows."""

        return self.observation_count


@dataclass(frozen=True)
class CausalOptionWindow:
    """Immutable causal European-call window and matched BS contract."""

    path_id: str
    path_seed: int
    regime: str
    split: str
    window_id: str
    start_index: int
    end_index: int
    prices: np.ndarray
    strike: float
    premium: float
    inception_volatility: float
    risk_free_rate: float
    maturity_years: float
    transaction_cost_convention_id: str
    liquidity: np.ndarray
    rolling_volatility: np.ndarray
    chartist_share: np.ndarray
    aggregate_demand: np.ndarray
    stress_state: np.ndarray
    noise_multiplier: np.ndarray
    black_scholes_delta_path: np.ndarray
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_text(self.path_id, "path_id")
        _require_nonempty_text(self.regime, "regime")
        _require_nonempty_text(self.window_id, "window_id")
        _require_nonempty_text(
            self.transaction_cost_convention_id,
            "transaction_cost_convention_id",
        )
        if not _is_plain_integer(self.path_seed):
            raise CausalDataValidationError("path_seed must be an integer label.")
        if self.split not in ALLOWED_SPLITS:
            raise CausalDataValidationError(
                f"split must be one of {ALLOWED_SPLITS}."
            )
        if not _is_plain_integer(self.start_index) or not _is_plain_integer(
            self.end_index
        ):
            raise CausalDataValidationError(
                "start_index and end_index must be integers."
            )
        if self.start_index < 63 or self.end_index != self.start_index + 63:
            raise CausalDataValidationError(
                "Window indices violate the canonical start/maturity contract."
            )
        expected_window_id = (
            f"{self.path_id}:{self.start_index}:{self.end_index}"
        )
        if self.window_id != expected_window_id:
            raise CausalDataValidationError(
                f"window_id must be {expected_window_id!r}."
            )

        prices = _readonly_float64_array(
            self.prices, field_name="prices", expected_length=64
        )
        if np.any(prices <= 0.0):
            raise CausalDataValidationError("prices must be strictly positive.")
        decision_arrays: Dict[str, np.ndarray] = {}
        for field_name in CANONICAL_FEATURE_ORDER:
            decision_arrays[field_name] = _readonly_float64_array(
                getattr(self, field_name),
                field_name=field_name,
                expected_length=63,
            )
        delta_path = _readonly_float64_array(
            self.black_scholes_delta_path,
            field_name="black_scholes_delta_path",
            expected_length=63,
        )
        if np.any(delta_path < -1e-12) or np.any(delta_path > 1.0 + 1e-12):
            raise CausalDataValidationError(
                "black_scholes_delta_path must remain inside [0, 1]."
            )

        strike = _require_finite_scalar(self.strike, "strike")
        premium = _require_finite_scalar(self.premium, "premium")
        sigma = _require_finite_scalar(
            self.inception_volatility, "inception_volatility"
        )
        risk_free_rate = _require_finite_scalar(
            self.risk_free_rate, "risk_free_rate"
        )
        maturity_years = _require_finite_scalar(
            self.maturity_years, "maturity_years"
        )
        if strike <= 0.0 or not math.isclose(
            strike, float(prices[0]), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise CausalDataValidationError(
                "strike must equal the positive start spot."
            )
        if premium < 0.0:
            raise CausalDataValidationError("premium cannot be negative.")
        if sigma < 1e-6:
            raise CausalDataValidationError(
                "inception_volatility is below the canonical floor."
            )
        if not math.isclose(
            maturity_years, 63.0 / 252.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise CausalDataValidationError(
                "maturity_years must equal 63/252."
            )
        if not math.isfinite(risk_free_rate):
            raise CausalDataValidationError("risk_free_rate must be finite.")
        if self.schema_version != SCHEMA_VERSION:
            raise CausalDataValidationError(
                f"schema_version must be {SCHEMA_VERSION!r}."
            )

        object.__setattr__(self, "prices", prices)
        for field_name, array in decision_arrays.items():
            object.__setattr__(self, field_name, array)
        object.__setattr__(self, "black_scholes_delta_path", delta_path)
        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "premium", premium)
        object.__setattr__(self, "inception_volatility", sigma)
        object.__setattr__(self, "risk_free_rate", risk_free_rate)
        object.__setattr__(self, "maturity_years", maturity_years)


def _resolve_config(config: Optional[CausalWindowConfig]) -> CausalWindowConfig:
    if config is None:
        return CausalWindowConfig()
    if not isinstance(config, CausalWindowConfig):
        raise CausalDataValidationError(
            "config must be a CausalWindowConfig instance."
        )
    return config


def validate_causal_path_frame(
    frame: pd.DataFrame,
    *,
    config: Optional[CausalWindowConfig] = None,
) -> pd.DataFrame:
    """Validate one path and return a deep copy with causal log returns.

    The returned frame has float64 numeric columns.  Its only permitted
    non-finite value is the required ``log_return`` sentinel at index zero.
    """

    _resolve_config(config)
    if not isinstance(frame, pd.DataFrame):
        raise CausalDataValidationError("frame must be a pandas DataFrame.")
    if len(frame) < 2:
        raise CausalDataValidationError("A causal path needs at least two prices.")
    if isinstance(frame.index, pd.MultiIndex):
        raise CausalDataValidationError("A path index cannot be a MultiIndex.")
    if not frame.index.is_unique:
        raise CausalDataValidationError("Path index values must be unique.")
    if not pd.api.types.is_integer_dtype(frame.index.dtype):
        raise CausalDataValidationError("Path index must have integer dtype.")
    expected_index = np.arange(len(frame), dtype=np.int64)
    observed_index = frame.index.to_numpy(dtype=np.int64, copy=True)
    if not np.array_equal(observed_index, expected_index):
        raise CausalDataValidationError(
            "Path index must be contiguous, strictly increasing and zero-based."
        )

    missing = [name for name in REQUIRED_PATH_COLUMNS if name not in frame.columns]
    if missing:
        raise CausalDataValidationError(
            "Missing required path columns: " + ", ".join(missing)
        )

    output = frame.copy(deep=True)
    numeric_columns: Dict[str, np.ndarray] = {}
    for column in REQUIRED_PATH_COLUMNS:
        numeric_columns[column] = _readonly_float64_array(
            frame[column].to_numpy(copy=True),
            field_name=column,
            expected_length=len(frame),
        )
    if np.any(numeric_columns["price"] <= 0.0):
        raise CausalDataValidationError("price must be strictly positive.")

    prices = numeric_columns["price"]
    derived_returns = np.empty(len(prices), dtype=np.float64)
    derived_returns[0] = np.nan
    derived_returns[1:] = np.log(prices[1:] / prices[:-1])
    if not np.isfinite(derived_returns[1:]).all():
        raise CausalDataValidationError("Derived log returns are not finite.")

    if "log_return" in frame.columns:
        supplied = _readonly_float64_array(
            frame["log_return"].to_numpy(copy=True),
            field_name="log_return",
            expected_length=len(frame),
            allow_nan=True,
        )
        if not np.isnan(supplied[0]):
            raise CausalDataValidationError(
                "Input log_return at index zero must be the NaN sentinel."
            )
        if np.isnan(supplied[1:]).any() or not np.isfinite(supplied[1:]).all():
            raise CausalDataValidationError(
                "Input log_return contains invalid values after index zero."
            )
        if not np.allclose(
            supplied[1:],
            derived_returns[1:],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        ):
            maximum_difference = float(
                np.max(np.abs(supplied[1:] - derived_returns[1:]))
            )
            raise CausalDataValidationError(
                "Input log_return does not match prices within the formula "
                f"tolerance; maximum difference={maximum_difference}."
            )

    for column, values in numeric_columns.items():
        output[column] = np.array(values, dtype=np.float64, copy=True)
    output["log_return"] = derived_returns
    if not np.isnan(float(output.loc[0, "log_return"])):
        raise CausalDataValidationError("Derived r_0 sentinel is missing.")
    return output


def compute_causal_rolling_volatility(
    frame: pd.DataFrame,
    *,
    config: Optional[CausalWindowConfig] = None,
) -> np.ndarray:
    """Return price-aligned trailing-only local volatility.

    Values at indices 0 through 19 are NaN.  Index 20 is the first defined
    value and uses exactly r_1 through r_20 with sample ``ddof=1``.
    """

    resolved = _resolve_config(config)
    validated = validate_causal_path_frame(frame, config=resolved)
    returns = validated["log_return"].to_numpy(dtype=np.float64, copy=True)
    result = np.full(len(returns), np.nan, dtype=np.float64)
    lookback = resolved.rolling_lookback
    for time_index in range(lookback, len(returns)):
        sample = returns[time_index - lookback + 1 : time_index + 1]
        if len(sample) != lookback or not np.isfinite(sample).all():
            raise CausalDataValidationError(
                f"Rolling sample is invalid at index {time_index}."
            )
        value = float(np.std(sample, ddof=1, dtype=np.float64))
        if not math.isfinite(value):
            raise CausalDataValidationError(
                f"Rolling volatility is invalid at index {time_index}."
            )
        result[time_index] = value
    result.setflags(write=False)
    return result


def _estimate_sigma_from_returns(
    returns: np.ndarray,
    start_index: int,
    config: CausalWindowConfig,
) -> float:
    if not _is_plain_integer(start_index):
        raise CausalDataValidationError("start_index must be an integer.")
    start = int(start_index)
    if start < config.inception_lookback:
        raise CausalDataValidationError(
            "Insufficient observed-return history for inception volatility."
        )
    if start >= len(returns):
        raise CausalDataValidationError("start_index is outside the path.")
    first = start - config.inception_lookback + 1
    sample = returns[first : start + 1]
    if len(sample) != config.inception_lookback or not np.isfinite(sample).all():
        raise CausalDataValidationError(
            "Inception volatility requires exactly 63 finite observed returns."
        )
    daily_sigma = float(np.std(sample, ddof=1, dtype=np.float64))
    annualized = daily_sigma * math.sqrt(float(config.steps_per_year))
    if not math.isfinite(annualized) or annualized < 0.0:
        raise CausalDataValidationError("Inception volatility is invalid.")
    return float(max(annualized, config.volatility_floor))


def estimate_inception_volatility(
    validated_frame: pd.DataFrame,
    start_index: int,
    *,
    config: Optional[CausalWindowConfig] = None,
) -> float:
    """Estimate sigma from the 63 observed returns ending at ``start_index``."""

    resolved = _resolve_config(config)
    if not isinstance(validated_frame, pd.DataFrame) or "log_return" not in validated_frame:
        raise CausalDataValidationError(
            "estimate_inception_volatility requires a validated frame with log_return."
        )
    checked = validate_causal_path_frame(validated_frame, config=resolved)
    returns = checked["log_return"].to_numpy(dtype=np.float64, copy=True)
    return _estimate_sigma_from_returns(returns, start_index, resolved)


def build_causal_black_scholes_delta_path(
    prices: Sequence[float],
    *,
    strike: float,
    inception_volatility: float,
    risk_free_rate: float,
    steps_per_year: int = 252,
    maturity_steps: int = 63,
) -> np.ndarray:
    """Build the fixed-inception-sigma 63-decision Black--Scholes delta path."""

    if not _is_plain_integer(steps_per_year) or int(steps_per_year) != 252:
        raise CausalDataValidationError("steps_per_year is frozen at 252.")
    if not _is_plain_integer(maturity_steps) or int(maturity_steps) != 63:
        raise CausalDataValidationError("maturity_steps is frozen at 63.")
    price_array = _readonly_float64_array(
        prices, field_name="prices", expected_length=int(maturity_steps) + 1
    )
    if np.any(price_array <= 0.0):
        raise CausalDataValidationError("prices must be strictly positive.")
    strike_value = _require_finite_scalar(strike, "strike")
    sigma = _require_finite_scalar(
        inception_volatility, "inception_volatility"
    )
    rate = _require_finite_scalar(risk_free_rate, "risk_free_rate")
    if strike_value <= 0.0 or sigma < 1e-6:
        raise CausalDataValidationError(
            "strike must be positive and inception_volatility must respect the floor."
        )
    deltas = np.empty(int(maturity_steps), dtype=np.float64)
    for relative_index in range(int(maturity_steps)):
        tau = float((int(maturity_steps) - relative_index) / int(steps_per_year))
        value = float(
            call_delta(
                S=float(price_array[relative_index]),
                K=strike_value,
                r=rate,
                sigma=sigma,
                tau=tau,
            )
        )
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise CausalDataValidationError(
                f"Black--Scholes delta is invalid at relative index {relative_index}."
            )
        deltas[relative_index] = value
    deltas.setflags(write=False)
    return deltas


def build_causal_option_windows(
    frame: pd.DataFrame,
    *,
    path_id: str,
    path_seed: int,
    regime: str,
    split: str,
    risk_free_rate: float,
    transaction_cost_convention_id: str,
    config: Optional[CausalWindowConfig] = None,
) -> Tuple[CausalOptionWindow, ...]:
    """Construct stable causal option windows from one already assigned path."""

    resolved = _resolve_config(config)
    _require_nonempty_text(path_id, "path_id")
    _require_nonempty_text(regime, "regime")
    _require_nonempty_text(
        transaction_cost_convention_id, "transaction_cost_convention_id"
    )
    if not _is_plain_integer(path_seed):
        raise CausalDataValidationError("path_seed must be an integer label.")
    if split not in ALLOWED_SPLITS:
        raise CausalDataValidationError(
            f"split must be one of {ALLOWED_SPLITS}; observed {split!r}."
        )
    rate = _require_finite_scalar(risk_free_rate, "risk_free_rate")
    validated = validate_causal_path_frame(frame, config=resolved)
    rolling = compute_causal_rolling_volatility(validated, config=resolved)
    starts = resolved.canonical_start_indices(len(validated))
    if not starts:
        raise CausalDataValidationError(
            "Path has insufficient history and horizon for one causal option window."
        )
    returns = validated["log_return"].to_numpy(dtype=np.float64, copy=True)
    windows = []
    for start in starts:
        end = start + resolved.maturity_steps
        prices = validated["price"].iloc[start : end + 1].to_numpy(
            dtype=np.float64, copy=True
        )
        if len(prices) != resolved.maturity_steps + 1:
            raise CausalDataValidationError(
                f"Window geometry failed at start index {start}."
            )
        decision_slice = slice(start, end)
        feature_arrays: Dict[str, np.ndarray] = {
            "rolling_volatility": np.array(
                rolling[decision_slice], dtype=np.float64, copy=True
            )
        }
        for feature_name in CANONICAL_FEATURE_ORDER:
            if feature_name == "rolling_volatility":
                continue
            feature_arrays[feature_name] = validated[feature_name].iloc[
                decision_slice
            ].to_numpy(dtype=np.float64, copy=True)
        for feature_name, values in feature_arrays.items():
            if len(values) != resolved.maturity_steps or not np.isfinite(values).all():
                raise CausalDataValidationError(
                    f"Decision feature {feature_name} is invalid at start {start}."
                )
        sigma = _estimate_sigma_from_returns(returns, start, resolved)
        strike = float(prices[0])
        premium = float(
            european_call_price(
                S=strike,
                K=strike,
                r=rate,
                sigma=sigma,
                tau=resolved.maturity_years,
            )
        )
        if not math.isfinite(premium) or premium < 0.0:
            raise CausalDataValidationError(
                f"Option premium is invalid at start index {start}."
            )
        deltas = build_causal_black_scholes_delta_path(
            prices,
            strike=strike,
            inception_volatility=sigma,
            risk_free_rate=rate,
            steps_per_year=resolved.steps_per_year,
            maturity_steps=resolved.maturity_steps,
        )
        window_id = f"{path_id}:{start}:{end}"
        windows.append(
            CausalOptionWindow(
                path_id=path_id,
                path_seed=int(path_seed),
                regime=regime,
                split=split,
                window_id=window_id,
                start_index=start,
                end_index=end,
                prices=prices,
                strike=strike,
                premium=premium,
                inception_volatility=sigma,
                risk_free_rate=rate,
                maturity_years=resolved.maturity_years,
                transaction_cost_convention_id=transaction_cost_convention_id,
                liquidity=feature_arrays["liquidity"],
                rolling_volatility=feature_arrays["rolling_volatility"],
                chartist_share=feature_arrays["chartist_share"],
                aggregate_demand=feature_arrays["aggregate_demand"],
                stress_state=feature_arrays["stress_state"],
                noise_multiplier=feature_arrays["noise_multiplier"],
                black_scholes_delta_path=deltas,
                schema_version=resolved.schema_version,
            )
        )
    expected_window_ids = sorted(window.window_id for window in windows)
    if len(expected_window_ids) != len(set(expected_window_ids)):
        raise CausalDataValidationError("Constructed window IDs are not unique.")
    return tuple(windows)


PartitionRecord = Union[CausalPathPartition, Mapping[str, object]]
WindowRecord = Union[CausalOptionWindow, Mapping[str, object]]


def _partition_from_record(record: PartitionRecord) -> CausalPathPartition:
    if isinstance(record, CausalPathPartition):
        return record
    if not isinstance(record, Mapping):
        raise CausalDataValidationError(
            "Partition records must be mappings or CausalPathPartition objects."
        )
    required = ("path_id", "path_seed", "regime", "split")
    missing = [name for name in required if name not in record]
    if missing:
        raise CausalDataValidationError(
            "Partition record is missing: " + ", ".join(missing)
        )
    return CausalPathPartition(
        path_id=str(record["path_id"]),
        path_seed=record["path_seed"],  # type: ignore[arg-type]
        regime=str(record["regime"]),
        split=str(record["split"]),
    )


def assign_path_partitions(
    path_records: Sequence[Mapping[str, object]],
    assignments: Mapping[str, str],
) -> Tuple[CausalPathPartition, ...]:
    """Apply an explicit path-id to role mapping without randomness."""

    if not path_records:
        raise CausalDataValidationError("path_records cannot be empty.")
    if not isinstance(assignments, Mapping) or not assignments:
        raise CausalDataValidationError("assignments must be a non-empty mapping.")
    seen_path_ids = set()
    normalized = []
    for record in path_records:
        if not isinstance(record, Mapping):
            raise CausalDataValidationError("Every path record must be a mapping.")
        missing = [name for name in ("path_id", "path_seed", "regime") if name not in record]
        if missing:
            raise CausalDataValidationError(
                "Path record is missing: " + ", ".join(missing)
            )
        path_id = _require_nonempty_text(record["path_id"], "path_id")
        if path_id in seen_path_ids:
            raise CausalDataValidationError(f"Duplicate path_id: {path_id!r}.")
        seen_path_ids.add(path_id)
        if path_id not in assignments:
            raise CausalDataValidationError(
                f"No explicit split assignment exists for {path_id!r}."
            )
        normalized.append(
            CausalPathPartition(
                path_id=path_id,
                path_seed=record["path_seed"],  # type: ignore[arg-type]
                regime=str(record["regime"]),
                split=str(assignments[path_id]),
            )
        )
    extra_assignments = sorted(set(assignments) - seen_path_ids)
    if extra_assignments:
        raise CausalDataValidationError(
            "Assignments contain unknown path IDs: " + ", ".join(extra_assignments)
        )
    result = tuple(normalized)
    validate_partition_isolation(result)
    return result


def _window_fields(record: WindowRecord) -> Dict[str, object]:
    names = ("path_id", "path_seed", "regime", "split", "window_id")
    if isinstance(record, CausalOptionWindow):
        return {name: getattr(record, name) for name in names}
    if not isinstance(record, Mapping):
        raise CausalDataValidationError(
            "Window records must be mappings or CausalOptionWindow objects."
        )
    missing = [name for name in names if name not in record]
    if missing:
        raise CausalDataValidationError(
            "Window record is missing: " + ", ".join(missing)
        )
    return {name: record[name] for name in names}


def validate_partition_isolation(
    partition_records: Sequence[PartitionRecord],
    windows: Sequence[WindowRecord] = (),
) -> Dict[str, object]:
    """Validate path-ID, path-seed, role and optional window isolation.

    A passing deterministic report is returned.  Any collision is blocking and
    raises ``CausalDataValidationError`` carrying the failed report.
    """

    if not partition_records:
        raise CausalDataValidationError("partition_records cannot be empty.")
    if not windows and all(
        isinstance(record, CausalOptionWindow) for record in partition_records
    ):
        windows = tuple(partition_records)  # type: ignore[assignment]
        unique_partitions: Dict[Tuple[str, int, str, str], CausalPathPartition] = {}
        for window in windows:
            assert isinstance(window, CausalOptionWindow)
            key = (
                window.path_id,
                int(window.path_seed),
                window.regime,
                window.split,
            )
            unique_partitions[key] = CausalPathPartition(
                path_id=window.path_id,
                path_seed=int(window.path_seed),
                regime=window.regime,
                split=window.split,
            )
        partition_records = tuple(unique_partitions.values())
    partitions = [_partition_from_record(record) for record in partition_records]
    path_id_roles: Dict[str, set] = {}
    seed_roles: Dict[int, set] = {}
    partition_by_id: Dict[str, CausalPathPartition] = {}
    duplicate_path_ids = []
    for partition in partitions:
        path_id_roles.setdefault(partition.path_id, set()).add(partition.split)
        seed_roles.setdefault(int(partition.path_seed), set()).add(partition.split)
        if partition.path_id in partition_by_id:
            duplicate_path_ids.append(partition.path_id)
        else:
            partition_by_id[partition.path_id] = partition
    path_id_collisions = sorted(
        set(duplicate_path_ids)
        | {
            path_id
            for path_id, roles in path_id_roles.items()
            if len(roles) > 1
        }
    )
    seed_collisions = sorted(
        seed for seed, roles in seed_roles.items() if len(roles) > 1
    )

    window_ids = set()
    window_id_collisions = []
    window_partition_mismatches = []
    for record in windows:
        values = _window_fields(record)
        window_id = str(values["window_id"])
        if window_id in window_ids:
            window_id_collisions.append(window_id)
        window_ids.add(window_id)
        path_id = str(values["path_id"])
        partition = partition_by_id.get(path_id)
        if partition is None:
            window_partition_mismatches.append(f"{window_id}:unknown_path_id")
            continue
        observed_seed = values["path_seed"]
        if not _is_plain_integer(observed_seed):
            window_partition_mismatches.append(f"{window_id}:invalid_path_seed")
            continue
        mismatch_fields = []
        if int(observed_seed) != int(partition.path_seed):
            mismatch_fields.append("path_seed")
        if str(values["regime"]) != partition.regime:
            mismatch_fields.append("regime")
        if str(values["split"]) != partition.split:
            mismatch_fields.append("split")
        if mismatch_fields:
            window_partition_mismatches.append(
                f"{window_id}:" + "+".join(mismatch_fields)
            )

    path_ids_by_role = {
        role: sorted(
            partition.path_id
            for partition in partitions
            if partition.split == role
        )
        for role in ALLOWED_SPLITS
    }
    path_seeds_by_role = {
        role: sorted(
            {
                int(partition.path_seed)
                for partition in partitions
                if partition.split == role
            }
        )
        for role in ALLOWED_SPLITS
    }
    failed = bool(
        path_id_collisions
        or seed_collisions
        or window_id_collisions
        or window_partition_mismatches
    )
    report: Dict[str, object] = {
        "status": "FAIL" if failed else "PASS",
        "partition_count": len(partitions),
        "window_count": len(windows),
        "roles": list(ALLOWED_SPLITS),
        "path_ids_by_role": path_ids_by_role,
        "path_seeds_by_role": path_seeds_by_role,
        "path_id_collisions": path_id_collisions,
        "path_id_cross_role_count": len(path_id_collisions),
        "path_seed_cross_role_collisions": seed_collisions,
        "path_seed_cross_role_count": len(seed_collisions),
        "window_id_collisions": sorted(set(window_id_collisions)),
        "window_partition_mismatches": sorted(window_partition_mismatches),
        "cross_role_overlap_count": 0,
    }
    if failed:
        raise CausalDataValidationError(
            "Path partition isolation failed.", report=report
        )
    return report


def fit_training_only_scalers(
    windows: Sequence[CausalOptionWindow],
    *,
    feature_names: Sequence[str] = CANONICAL_FEATURE_ORDER,
) -> Mapping[str, ScalerStats]:
    """Fit immutable float64 population scalers using training windows only."""

    if not windows:
        raise CausalDataValidationError("Scaler input cannot be empty.")
    ordered_features = tuple(feature_names)
    if ordered_features != CANONICAL_FEATURE_ORDER:
        raise CausalDataValidationError(
            "feature_names must equal the frozen canonical feature order."
        )
    for window in windows:
        if not isinstance(window, CausalOptionWindow):
            raise CausalDataValidationError(
                "Scaler input must contain CausalOptionWindow objects."
            )
    validate_partition_isolation(tuple(windows))
    ordered_windows = tuple(
        sorted(
            (window for window in windows if window.split == "train"),
            key=lambda item: item.window_id,
        )
    )
    if not ordered_windows:
        raise CausalDataValidationError(
            "Scaler fitting requires at least one training window."
        )
    window_ids = tuple(window.window_id for window in ordered_windows)
    if len(window_ids) != len(set(window_ids)):
        raise CausalDataValidationError("Scaler window IDs must be unique.")

    result: Dict[str, ScalerStats] = {}
    for feature_name in ordered_features:
        values = np.concatenate(
            [
                np.asarray(getattr(window, feature_name), dtype=np.float64)
                for window in ordered_windows
            ]
        ).astype(np.float64, copy=False)
        if len(values) != 63 * len(ordered_windows):
            raise CausalDataValidationError(
                f"Scaler observation count is invalid for {feature_name}."
            )
        if not np.isfinite(values).all():
            raise CausalDataValidationError(
                f"Scaler feature {feature_name} contains non-finite values."
            )
        mean = float(np.mean(values, dtype=np.float64))
        standard_deviation = float(np.std(values, ddof=0, dtype=np.float64))
        if not math.isfinite(mean) or not math.isfinite(standard_deviation):
            raise CausalDataValidationError(
                f"Scaler statistic is non-finite for {feature_name}."
            )
        if standard_deviation <= 0.0:
            raise CausalDataValidationError(
                f"Scaler standard deviation is non-positive for {feature_name}."
            )
        result[feature_name] = ScalerStats(
            feature_name=feature_name,
            mean=mean,
            standard_deviation=standard_deviation,
            observation_count=len(values),
            ddof=0,
            training_window_ids=window_ids,
            schema_version=SCHEMA_VERSION,
        )
    return MappingProxyType(result)


def hash_causal_array(field_name: str, values: Sequence[float]) -> str:
    """Hash field name, explicit dtype/shape and C-order float64 bytes."""

    name = _require_nonempty_text(field_name, "field_name")
    array = _readonly_float64_array(values, field_name=name)
    metadata = {
        "dtype": str(array.dtype),
        "field_name": name,
        "order": "C",
        "shape": list(array.shape),
    }
    metadata_bytes = (
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = hashlib.sha256()
    digest.update(metadata_bytes)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def canonical_window_record(window: CausalOptionWindow) -> Dict[str, object]:
    """Return a deterministic JSON-compatible core record for one window."""

    if not isinstance(window, CausalOptionWindow):
        raise CausalDataValidationError(
            "window must be a CausalOptionWindow instance."
        )
    array_names = (
        "prices",
        "liquidity",
        "rolling_volatility",
        "chartist_share",
        "aggregate_demand",
        "stress_state",
        "noise_multiplier",
        "black_scholes_delta_path",
    )
    array_hashes = {
        name: hash_causal_array(name, getattr(window, name))
        for name in array_names
    }
    return {
        "schema_version": window.schema_version,
        "path_id": window.path_id,
        "path_seed": int(window.path_seed),
        "regime": window.regime,
        "split": window.split,
        "window_id": window.window_id,
        "start_index": int(window.start_index),
        "end_index": int(window.end_index),
        "strike": float(window.strike),
        "premium": float(window.premium),
        "inception_volatility": float(window.inception_volatility),
        "risk_free_rate": float(window.risk_free_rate),
        "maturity_years": float(window.maturity_years),
        "transaction_cost_convention_id": window.transaction_cost_convention_id,
        "array_hashes": array_hashes,
    }

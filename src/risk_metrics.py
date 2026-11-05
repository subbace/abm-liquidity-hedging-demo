import numpy as np
import pandas as pd


def clean_numeric_array(values):
    """
    Convert input values to a clean NumPy array.

    This function removes:
    - missing values
    - positive infinity
    - negative infinity
    """

    array = np.asarray(values, dtype=float)

    array = array[np.isfinite(array)]

    if len(array) == 0:
        raise ValueError("The input array contains no valid numeric values.")

    return array


def value_at_risk(losses, confidence_level=0.95):
    """
    Compute Value-at-Risk from a loss distribution.

    Convention
    ----------
    Losses are positive when the strategy loses money.

    For example, at confidence level 95%, VaR is the 95th percentile
    of the loss distribution.

    Parameters
    ----------
    losses : array-like
        Loss observations.
    confidence_level : float
        Confidence level, for example 0.95.

    Returns
    -------
    float
        Value-at-Risk.
    """

    losses = clean_numeric_array(losses)

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be inside (0, 1).")

    var = np.quantile(losses, confidence_level)

    return var


def expected_shortfall(losses, confidence_level=0.95):
    """
    Compute Expected Shortfall from a loss distribution.

    Convention
    ----------
    Losses are positive when the strategy loses money.

    Expected Shortfall is the average loss conditional on losses
    being larger than or equal to the Value-at-Risk.

    Parameters
    ----------
    losses : array-like
        Loss observations.
    confidence_level : float
        Confidence level, for example 0.95.

    Returns
    -------
    float
        Expected Shortfall.
    """

    losses = clean_numeric_array(losses)

    var = value_at_risk(
        losses=losses,
        confidence_level=confidence_level,
    )

    tail_losses = losses[losses >= var]

    if len(tail_losses) == 0:
        return var

    es = tail_losses.mean()

    return es


def compute_hedging_error_metrics(hedging_errors, confidence_level=0.95):
    """
    Compute summary statistics and risk metrics for hedging errors.

    Hedging error convention
    ------------------------
    The hedging error is defined as:

        HE = final portfolio value - option payoff

    If HE is positive, the hedge finishes with a surplus.
    If HE is negative, the hedge finishes with a deficit.

    Loss convention
    ---------------
    For risk metrics, losses are defined as:

        Loss = - HE

    Therefore:
    - negative hedging error means positive loss;
    - positive hedging error means negative loss or surplus.

    Parameters
    ----------
    hedging_errors : array-like
        Hedging error observations.
    confidence_level : float
        Confidence level for VaR and Expected Shortfall.

    Returns
    -------
    dict
        Dictionary with summary statistics.
    """

    errors = clean_numeric_array(hedging_errors)

    losses = -errors

    mean_error = errors.mean()
    standard_deviation = errors.std(ddof=1)
    mean_absolute_error = np.mean(np.abs(errors))
    root_mean_squared_error = np.sqrt(np.mean(errors**2))

    var = value_at_risk(
        losses=losses,
        confidence_level=confidence_level,
    )

    es = expected_shortfall(
        losses=losses,
        confidence_level=confidence_level,
    )

    metrics = {
        "count": len(errors),
        "mean_hedging_error": mean_error,
        "standard_deviation": standard_deviation,
        "mean_absolute_error": mean_absolute_error,
        "root_mean_squared_error": root_mean_squared_error,
        "minimum_hedging_error": errors.min(),
        "maximum_hedging_error": errors.max(),
        "var_95": var,
        "expected_shortfall_95": es,
    }

    return metrics


def compute_metrics_by_regime(results, regime_column="regime", error_column="hedging_error"):
    """
    Compute hedging error metrics separately for each market regime.

    Parameters
    ----------
    results : pandas.DataFrame
        DataFrame containing at least regime and hedging error columns.
    regime_column : str
        Name of the column containing regime labels.
    error_column : str
        Name of the column containing hedging errors.

    Returns
    -------
    pandas.DataFrame
        Summary table with one row per regime.
    """

    if regime_column not in results.columns:
        raise ValueError(f"Column not found: {regime_column}")

    if error_column not in results.columns:
        raise ValueError(f"Column not found: {error_column}")

    rows = []

    for regime, group in results.groupby(regime_column):
        metrics = compute_hedging_error_metrics(
            hedging_errors=group[error_column].values,
            confidence_level=0.95,
        )

        metrics["regime"] = regime
        rows.append(metrics)

    summary = pd.DataFrame(rows)

    columns = [
        "regime",
        "count",
        "mean_hedging_error",
        "standard_deviation",
        "mean_absolute_error",
        "root_mean_squared_error",
        "minimum_hedging_error",
        "maximum_hedging_error",
        "var_95",
        "expected_shortfall_95",
    ]

    summary = summary[columns]

    return summary
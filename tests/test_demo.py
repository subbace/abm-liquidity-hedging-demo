"""Portable tests on synthetic fixtures only; not the full thesis audit suite."""
from decimal import Decimal, localcontext
import unittest

import numpy as np
import pandas as pd

from run_demo import verify_preserved_sources
from src.abm_simulator import ABMSimulator
from src.causal_black_scholes_baseline import adapt_abm_v042_frame, simulate_causal_black_scholes_window
from src.causal_hedging_data import CausalDataValidationError, build_causal_option_windows
from src.risk_metrics import expected_shortfall, value_at_risk


def analytic_frame():
    t = np.arange(187, dtype=float)
    return pd.DataFrame({
        "price": 100 * np.exp(0.0003 * t + 0.001 * np.sin(t / 7)),
        "liquidity": 1.0 + 0.1 * np.sin(t / 11),
        "chartist_share": np.full(187, 0.5),
        "aggregate_demand": 0.02 * np.cos(t / 9),
        "stress_state": 0.2 + 0.01 * np.cos(t / 13),
        "noise_multiplier": 1.0 + 0.02 * np.sin(t / 5),
    })


def windows(frame):
    return build_causal_option_windows(
        frame, path_id="analytic", path_seed=11, regime="analytic",
        split="validation", risk_free_rate=0.02,
        transaction_cost_convention_id="synthetic_test_0.001",
    )


class DemoTests(unittest.TestCase):
    def test_preserved_source_provenance(self):
        self.assertEqual(len(verify_preserved_sources()["preserved_sources"]), 6)

    def test_same_seed_reproduces_small_synthetic_path(self):
        first = ABMSimulator(T=128, seed=11).run()
        second = ABMSimulator(T=128, seed=11).run()
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        different = ABMSimulator(T=128, seed=12).run()
        self.assertFalse(np.array_equal(first["price"], different["price"]))

    def test_zero_demand_liquidity_floor_has_known_solution(self):
        frame = ABMSimulator(T=8, phi=0, chi=0, sigma_N=0, L0=0.5,
                             liquidity_floor=0.7, seed=11).run()
        np.testing.assert_allclose(frame["aggregate_demand"].iloc[1:], 0, atol=0, rtol=0)
        np.testing.assert_allclose(frame["raw_liquidity"].iloc[1:], 0.5, atol=0, rtol=0)
        np.testing.assert_allclose(frame["liquidity"].iloc[1:], 0.7, atol=0, rtol=0)
        np.testing.assert_allclose(frame["price"], 100.0, atol=0, rtol=0)

    def test_future_changes_do_not_change_prior_decisions_or_sigma(self):
        original = analytic_frame()
        changed = original.copy(deep=True)
        changed.loc[85:, "price"] *= 0.97
        changed.loc[85:, "liquidity"] *= 1.2
        a, b = windows(original)[0], windows(changed)[0]
        self.assertEqual(a.inception_volatility, b.inception_volatility)
        self.assertEqual(a.premium, b.premium)
        before, after = simulate_causal_black_scholes_window(a), simulate_causal_black_scholes_window(b)
        np.testing.assert_array_equal(before.deltas[:22], after.deltas[:22])
        np.testing.assert_array_equal(before.cash_after_accrual[:22], after.cash_after_accrual[:22])
        self.assertNotEqual(before.deltas[22], after.deltas[22])

    def test_cash_reconciles_with_independent_decimal_recursion(self):
        result = simulate_causal_black_scholes_window(windows(analytic_frame())[0])
        with localcontext() as context:
            context.prec = 45
            dec = lambda x: Decimal(str(float(x)))
            cash, previous = dec(result.premium), Decimal(0)
            growth = (Decimal("0.02") / Decimal(252)).exp()
            fees = Decimal(0)
            for price, delta in zip(result.prices[:-1], result.deltas):
                trade = dec(delta) - previous
                fee = Decimal("0.001") * abs(trade) * dec(price)
                cash = (cash - trade * dec(price) - fee) * growth
                fees += fee
                previous = dec(delta)
            terminal = cash + previous * dec(result.prices[-1])
            payoff = max(dec(result.prices[-1]) - dec(result.strike), Decimal(0))
            self.assertAlmostEqual(float(terminal - payoff), result.hedging_error, delta=1e-10)
            self.assertAlmostEqual(float(fees), result.total_transaction_cost, delta=1e-12)
        self.assertEqual(len(result.trade_sizes), 63)
        self.assertAlmostEqual(result.transaction_costs[0], 0.001 * result.deltas[0] * result.prices[0], delta=1e-12)
        self.assertAlmostEqual(result.total_transaction_cost, 0.001 * result.notional_turnover, delta=1e-12)

    def test_invalid_prices_are_rejected(self):
        bad = analytic_frame()
        bad.loc[40, "price"] = -1
        with self.assertRaises(CausalDataValidationError):
            windows(bad)

    def test_original_full_path_adapter_guard_is_preserved(self):
        with self.assertRaisesRegex(CausalDataValidationError, "5,001"):
            adapt_abm_v042_frame(ABMSimulator(T=20, seed=11).run())

    def test_tail_metrics_on_hand_calculated_losses(self):
        self.assertEqual(value_at_risk([0, 1, 2, 3], confidence_level=0.5), 1.5)
        self.assertEqual(expected_shortfall([0, 1, 2, 3], confidence_level=0.5), 2.5)


if __name__ == "__main__":
    unittest.main()

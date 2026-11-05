import numpy as np
import pandas as pd


class ABMSimulator:
    """
    Agent-Based Market Simulator with persistent market stress.

    The model includes:
    - fundamentalist demand
    - chartist demand
    - noise-trader demand
    - optional short-horizon contrarian demand
    - optional sparse symmetric rare-shock demand
    - aggregate excess demand
    - endogenous liquidity
    - persistent market stress
    - liquidity-sensitive noise
    - price formation through liquidity-adjusted price impact
    - switching and herding dynamics between fundamentalists and chartists
    - optional herding bias to create stronger chartist-dominated regimes
    """

    def __init__(
        self,
        T=5000,
        P0=100.0,
        P_star=100.0,
        x0=0.5,
        phi=0.8,
        chi=1.2,
        sigma_N=0.02,
        short_horizon_reversal_strength=0.0,
        rare_shock_probability=0.0,
        rare_shock_scale=0.0,
        rare_shock_log_sigma=0.50,
        lambda_price=0.05,
        L0=1.0,
        gamma=2.0,
        liquidity_floor=0.10,
        liquidity_persistence=0.90,
        noise_liquidity_sensitivity=0.60,
        max_noise_multiplier=8.0,
        stress_demand_scale=0.04,
        stress_return_scale=0.006,
        stress_impact_on_liquidity=1.0,
        max_stress=5.0,
        max_abs_return=0.20,
        eta=0.001,
        h=0.02,
        beta=0.5,
        herding_bias=0.0,
        switching_function="linear_clipped",
        switching_temperature=0.05,
        switching_threshold=0.03,
        switching_scale=1.0,
        enable_liquidity_provider=False,
        lp_base_depth=1.0,
        lp_inventory_sensitivity=0.25,
        lp_volatility_sensitivity=0.50,
        lp_stress_sensitivity=0.75,
        lp_min_depth=0.20,
        lp_max_depth=3.00,
        lp_spread_base=0.001,
        lp_spread_volatility_sensitivity=0.50,
        lp_spread_stress_sensitivity=0.75,
        lp_impact_sensitivity=0.50,
        lp_max_impact_multiplier=2.00,
        lp_liquidity_supply_scale=0.30,
        lp_withdrawal_sensitivity=0.75,
        lp_max_liquidity_multiplier=1.50,
        lp_min_liquidity_multiplier=0.50,
        lp_inventory_mean_reversion=0.05,
        lp_inventory_limit=5.0,
        seed=42,
    ):
        self.T = T

        # Price and fundamental value
        self.P0 = P0
        self.P_star = P_star

        # Initial chartist share
        self.x0 = x0

        # Demand parameters
        self.phi = phi
        self.chi = chi
        self.sigma_N = sigma_N
        self.short_horizon_reversal_strength = (
            short_horizon_reversal_strength
        )
        self.rare_shock_probability = rare_shock_probability
        self.rare_shock_scale = rare_shock_scale
        self.rare_shock_log_sigma = rare_shock_log_sigma

        # Price impact and liquidity parameters
        self.lambda_price = lambda_price
        self.L0 = L0
        self.gamma = gamma
        self.liquidity_floor = liquidity_floor

        # Persistent stress parameters
        self.liquidity_persistence = liquidity_persistence
        self.noise_liquidity_sensitivity = noise_liquidity_sensitivity
        self.max_noise_multiplier = max_noise_multiplier
        self.stress_demand_scale = stress_demand_scale
        self.stress_return_scale = stress_return_scale
        self.stress_impact_on_liquidity = stress_impact_on_liquidity
        self.max_stress = max_stress
        self.max_abs_return = max_abs_return

        # Switching parameters
        self.eta = eta
        self.h = h
        self.beta = beta
        self.herding_bias = herding_bias
        self.switching_function = switching_function
        self.switching_temperature = switching_temperature
        self.switching_threshold = switching_threshold
        self.switching_scale = switching_scale

        # Optional liquidity-provider / market-maker extension.
        self.enable_liquidity_provider = enable_liquidity_provider
        self.lp_base_depth = lp_base_depth
        self.lp_inventory_sensitivity = lp_inventory_sensitivity
        self.lp_volatility_sensitivity = lp_volatility_sensitivity
        self.lp_stress_sensitivity = lp_stress_sensitivity
        self.lp_min_depth = lp_min_depth
        self.lp_max_depth = lp_max_depth
        self.lp_spread_base = lp_spread_base
        self.lp_spread_volatility_sensitivity = lp_spread_volatility_sensitivity
        self.lp_spread_stress_sensitivity = lp_spread_stress_sensitivity
        self.lp_impact_sensitivity = lp_impact_sensitivity
        self.lp_max_impact_multiplier = lp_max_impact_multiplier
        self.lp_liquidity_supply_scale = lp_liquidity_supply_scale
        self.lp_withdrawal_sensitivity = lp_withdrawal_sensitivity
        self.lp_max_liquidity_multiplier = lp_max_liquidity_multiplier
        self.lp_min_liquidity_multiplier = lp_min_liquidity_multiplier
        self.lp_inventory_mean_reversion = lp_inventory_mean_reversion
        self.lp_inventory_limit = lp_inventory_limit

        # Random seed
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.rare_shock_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 1301])
        )

        self._validate_parameters()

    def _validate_parameters(self):
        """
        Basic parameter checks.
        """

        if self.T <= 0:
            raise ValueError("T must be positive.")

        if self.P0 <= 0:
            raise ValueError("P0 must be positive.")

        if self.P_star <= 0:
            raise ValueError("P_star must be positive.")

        if not 0.0 <= self.x0 <= 1.0:
            raise ValueError("x0 must be inside [0, 1].")

        if self.short_horizon_reversal_strength < 0:
            raise ValueError(
                "short_horizon_reversal_strength cannot be negative."
            )

        if not 0.0 <= self.rare_shock_probability <= 1.0:
            raise ValueError(
                "rare_shock_probability must be inside [0, 1]."
            )

        if self.rare_shock_scale < 0:
            raise ValueError(
                "rare_shock_scale cannot be negative."
            )

        if self.rare_shock_log_sigma < 0:
            raise ValueError(
                "rare_shock_log_sigma cannot be negative."
            )

        if self.L0 <= 0:
            raise ValueError("L0 must be positive.")

        if self.liquidity_floor <= 0:
            raise ValueError("liquidity_floor must be positive.")

        if not 0.0 <= self.liquidity_persistence < 1.0:
            raise ValueError("liquidity_persistence must be inside [0, 1).")

        if self.max_noise_multiplier <= 0:
            raise ValueError("max_noise_multiplier must be positive.")

        if self.stress_demand_scale <= 0:
            raise ValueError("stress_demand_scale must be positive.")

        if self.stress_return_scale <= 0:
            raise ValueError("stress_return_scale must be positive.")

        if self.max_stress <= 0:
            raise ValueError("max_stress must be positive.")

        if self.max_abs_return <= 0:
            raise ValueError("max_abs_return must be positive.")

        if not -0.005 <= self.herding_bias <= 0.005:
            raise ValueError("herding_bias must be inside [-0.005, 0.005].")

        valid_switching_functions = {
            "linear_clipped",
            "logistic_switching",
            "threshold_switching",
        }
        if self.switching_function not in valid_switching_functions:
            raise ValueError(
                "switching_function must be one of: "
                + ", ".join(sorted(valid_switching_functions))
            )

        if self.switching_temperature <= 0:
            raise ValueError("switching_temperature must be positive.")

        if self.switching_threshold < 0:
            raise ValueError("switching_threshold cannot be negative.")

        if self.switching_scale <= 0:
            raise ValueError("switching_scale must be positive.")

        if self.lp_base_depth <= 0:
            raise ValueError("lp_base_depth must be positive.")

        if self.lp_min_depth <= 0:
            raise ValueError("lp_min_depth must be positive.")

        if self.lp_max_depth < self.lp_min_depth:
            raise ValueError("lp_max_depth must be >= lp_min_depth.")

        if self.lp_inventory_limit <= 0:
            raise ValueError("lp_inventory_limit must be positive.")

        if self.lp_max_impact_multiplier < 1.0:
            raise ValueError("lp_max_impact_multiplier must be >= 1.")

        if self.lp_liquidity_supply_scale < 0:
            raise ValueError("lp_liquidity_supply_scale cannot be negative.")

        if self.lp_withdrawal_sensitivity < 0:
            raise ValueError("lp_withdrawal_sensitivity cannot be negative.")

        if self.lp_max_liquidity_multiplier < 1.0:
            raise ValueError("lp_max_liquidity_multiplier must be >= 1.")

        if not 0.0 < self.lp_min_liquidity_multiplier <= 1.0:
            raise ValueError("lp_min_liquidity_multiplier must be inside (0, 1].")

        if self.lp_min_liquidity_multiplier > self.lp_max_liquidity_multiplier:
            raise ValueError(
                "lp_min_liquidity_multiplier must be <= "
                "lp_max_liquidity_multiplier."
            )

        if not 0.0 <= self.lp_inventory_mean_reversion < 1.0:
            raise ValueError(
                "lp_inventory_mean_reversion must be inside [0, 1)."
            )

        non_negative_lp_parameters = {
            "lp_inventory_sensitivity": self.lp_inventory_sensitivity,
            "lp_volatility_sensitivity": self.lp_volatility_sensitivity,
            "lp_stress_sensitivity": self.lp_stress_sensitivity,
            "lp_spread_base": self.lp_spread_base,
            "lp_spread_volatility_sensitivity": (
                self.lp_spread_volatility_sensitivity
            ),
            "lp_spread_stress_sensitivity": self.lp_spread_stress_sensitivity,
            "lp_impact_sensitivity": self.lp_impact_sensitivity,
        }
        for parameter_name, parameter_value in non_negative_lp_parameters.items():
            if parameter_value < 0:
                raise ValueError(f"{parameter_name} cannot be negative.")

    @staticmethod
    def _clip_probability(value):
        """
        Probabilities must remain inside [0, 1].
        """
        return np.clip(value, 0.0, 1.0)

    @staticmethod
    def _sigmoid(value):
        """
        Numerically stable logistic transform.
        """
        value = np.clip(value, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-value))

    def _compute_switching_probabilities(
        self,
        previous_chartist_share,
        previous_return,
    ):
        """
        Compute probabilities of switching between fundamentalist and
        chartist behavior.

        The default linear_clipped specification reproduces the original
        ABM v0.4.2 mechanism exactly. Alternative specifications are used
        only when explicitly requested by an experiment script.
        """

        f_to_c_signal = (
            self.h * previous_chartist_share
            + self.beta * max(previous_return, 0.0)
            + self.herding_bias
        )

        c_to_f_signal = (
            self.h * (1.0 - previous_chartist_share)
            + self.beta * max(-previous_return, 0.0)
            - self.herding_bias
        )

        if self.switching_function == "linear_clipped":
            p_f_to_c = self.eta + f_to_c_signal
            p_c_to_f = self.eta + c_to_f_signal

        elif self.switching_function == "logistic_switching":
            # Smooth alternative to direct clipping. The centered logistic
            # term is zero when the behavioral signal is zero, so eta remains
            # the baseline spontaneous switching probability.
            amplitude = 0.08 * self.switching_scale
            p_f_to_c = self.eta + amplitude * (
                2.0 * self._sigmoid(
                    f_to_c_signal / self.switching_temperature
                )
                - 1.0
            )
            p_c_to_f = self.eta + amplitude * (
                2.0 * self._sigmoid(
                    c_to_f_signal / self.switching_temperature
                )
                - 1.0
            )

        elif self.switching_function == "threshold_switching":
            # Weak switching under normal conditions, with stronger response
            # only after the momentum/herding signal crosses a threshold.
            p_f_to_c = (
                self.eta
                + 0.25 * f_to_c_signal
                + self.switching_scale
                * max(f_to_c_signal - self.switching_threshold, 0.0)
            )
            p_c_to_f = (
                self.eta
                + 0.25 * c_to_f_signal
                + self.switching_scale
                * max(c_to_f_signal - self.switching_threshold, 0.0)
            )

        return (
            self._clip_probability(p_f_to_c),
            self._clip_probability(p_c_to_f),
        )

    def _compute_liquidity_provider_state(
        self,
        baseline_liquidity,
        aggregate_demand,
        previous_return,
        stress_state,
        previous_inventory,
    ):
        """
        Compute a minimal market-maker liquidity supply layer.

        Positive aggregate demand means net buying pressure from non-market
        makers. The liquidity provider sells into that demand, so inventory
        decreases. Inventory mean-reverts slowly between periods.
        """

        inventory = (
            (1.0 - self.lp_inventory_mean_reversion) * previous_inventory
            - aggregate_demand
        )
        inventory = np.clip(
            inventory,
            -self.lp_inventory_limit,
            self.lp_inventory_limit,
        )

        inventory_pressure = min(
            abs(inventory) / self.lp_inventory_limit,
            1.0,
        )
        volatility_pressure = min(
            abs(previous_return) / max(self.stress_return_scale, 1e-12),
            self.max_stress,
        ) / self.max_stress
        stress_pressure = min(
            stress_state / max(self.max_stress, 1e-12),
            1.0,
        )
        imbalance_pressure = min(
            abs(aggregate_demand) / max(self.stress_demand_scale, 1e-12),
            self.max_stress,
        ) / self.max_stress

        depth_penalty = (
            self.lp_inventory_sensitivity * inventory_pressure
            + self.lp_volatility_sensitivity * volatility_pressure
            + self.lp_stress_sensitivity * stress_pressure
        )
        depth = self.lp_base_depth / (1.0 + depth_penalty)
        depth = np.clip(depth, self.lp_min_depth, self.lp_max_depth)

        depth_fraction = depth / max(self.lp_base_depth, 1e-12)

        withdrawal_pressure = min(
            1.0,
            0.35 * stress_pressure
            + 0.25 * volatility_pressure
            + 0.25 * inventory_pressure
            + 0.15 * imbalance_pressure,
        )

        liquidity_supply_multiplier = (
            1.0
            + self.lp_liquidity_supply_scale * depth_fraction
            - self.lp_withdrawal_sensitivity * withdrawal_pressure
        )
        liquidity_supply_multiplier = np.clip(
            liquidity_supply_multiplier,
            self.lp_min_liquidity_multiplier,
            self.lp_max_liquidity_multiplier,
        )

        effective_liquidity = baseline_liquidity * liquidity_supply_multiplier
        effective_liquidity = max(self.liquidity_floor, effective_liquidity)

        depth_shortfall = max(self.lp_base_depth / max(depth, 1e-12) - 1.0, 0.0)
        market_impact_multiplier = 1.0 + self.lp_impact_sensitivity * (
            0.25 * depth_shortfall
            + 0.35 * withdrawal_pressure
            + 0.20 * inventory_pressure
            + 0.20 * imbalance_pressure
        )
        market_impact_multiplier = min(
            market_impact_multiplier,
            self.lp_max_impact_multiplier,
        )

        spread = self.lp_spread_base * (
            1.0
            + self.lp_spread_volatility_sensitivity * volatility_pressure
            + self.lp_spread_stress_sensitivity * stress_pressure
            + self.lp_inventory_sensitivity * inventory_pressure
        )
        spread = np.clip(spread, 0.0, 0.25)

        transaction_cost_proxy = spread + (
            self.lambda_price
            * abs(aggregate_demand)
            / max(effective_liquidity, 1e-12)
            * market_impact_multiplier
        )

        return {
            "inventory": inventory,
            "inventory_pressure": inventory_pressure,
            "depth": depth,
            "spread": spread,
            "effective_liquidity": effective_liquidity,
            "market_impact_multiplier": market_impact_multiplier,
            "liquidity_supply_multiplier": liquidity_supply_multiplier,
            "withdrawal_pressure": withdrawal_pressure,
            "transaction_cost_proxy": transaction_cost_proxy,
        }

    def run(self):
        """
        Run the ABM simulation and return a pandas DataFrame
        containing all simulated variables.
        """

        # Main time series
        price = np.zeros(self.T + 1)
        log_return = np.zeros(self.T + 1)
        chartist_share = np.zeros(self.T + 1)

        # Demand components
        fundamentalist_demand = np.zeros(self.T + 1)
        chartist_demand = np.zeros(self.T + 1)
        noise_demand = np.zeros(self.T + 1)
        reversal_demand = np.zeros(self.T + 1)
        rare_shock_event = np.zeros(self.T + 1)
        rare_shock_sign = np.zeros(self.T + 1)
        rare_shock_magnitude = np.zeros(self.T + 1)
        rare_shock_demand = np.zeros(self.T + 1)
        aggregate_demand = np.zeros(self.T + 1)

        # Liquidity variables
        raw_liquidity = np.zeros(self.T + 1)
        liquidity = np.zeros(self.T + 1)

        # Optional liquidity-provider diagnostics. These columns are always
        # returned with neutral values when the extension is disabled.
        lp_inventory = np.zeros(self.T + 1)
        lp_inventory_pressure = np.zeros(self.T + 1)
        lp_depth = np.full(self.T + 1, self.lp_base_depth)
        lp_spread = np.zeros(self.T + 1)
        lp_effective_liquidity = np.zeros(self.T + 1)
        lp_market_impact_multiplier = np.ones(self.T + 1)
        lp_liquidity_supply_multiplier = np.ones(self.T + 1)
        lp_withdrawal_pressure = np.zeros(self.T + 1)
        lp_effective_transaction_cost_proxy = np.zeros(self.T + 1)

        # Persistent stress variables
        stress_state = np.zeros(self.T + 1)
        stress_target = np.zeros(self.T + 1)

        # Liquidity-sensitive noise variables
        noise_scale = np.zeros(self.T + 1)
        noise_multiplier = np.zeros(self.T + 1)

        # Switching probabilities
        p_F_to_C = np.zeros(self.T + 1)
        p_C_to_F = np.zeros(self.T + 1)

        # Initial conditions
        price[0] = self.P0
        chartist_share[0] = self.x0
        raw_liquidity[0] = self.L0
        liquidity[0] = self.L0
        lp_effective_liquidity[0] = self.L0
        if self.enable_liquidity_provider:
            lp_spread[0] = self.lp_spread_base
            lp_effective_transaction_cost_proxy[0] = self.lp_spread_base
        stress_state[0] = 0.0
        stress_target[0] = 0.0
        noise_scale[0] = self.sigma_N
        noise_multiplier[0] = 1.0

        for t in range(1, self.T + 1):

            previous_price = price[t - 1]
            previous_return = log_return[t - 1]
            previous_chartist_share = chartist_share[t - 1]
            previous_stress = stress_state[t - 1]

            # ------------------------------------------------------------
            # 1. Fundamentalist demand
            #
            # Formula:
            # D_F,t = phi * ((P_star - P_t) / P_t)
            # ------------------------------------------------------------
            fundamentalist_demand[t] = self.phi * (
                (self.P_star - previous_price) / previous_price
            )

            # ------------------------------------------------------------
            # 2. Chartist demand
            #
            # Formula:
            # D_C,t = chi * r_{t-1}
            # ------------------------------------------------------------
            chartist_demand[t] = self.chi * previous_return

            # ------------------------------------------------------------
            # 3. Liquidity-sensitive noise-trader demand
            #
            # The noise shock becomes larger when the persistent stress
            # state is high.
            #
            # sigma_{N,t} = sigma_N * min(
            #     max_noise_multiplier,
            #     1 + noise_liquidity_sensitivity * S_{t-1}
            # )
            # ------------------------------------------------------------
            noise_multiplier[t] = 1.0 + (
                self.noise_liquidity_sensitivity * previous_stress
            )

            noise_multiplier[t] = min(
                noise_multiplier[t],
                self.max_noise_multiplier
            )

            noise_scale[t] = self.sigma_N * noise_multiplier[t]

            noise_demand[t] = self.rng.normal(
                loc=0.0,
                scale=noise_scale[t]
            )

            # ------------------------------------------------------------
            # 3b. Optional short-horizon contrarian demand
            #
            # Formula:
            # D_REV,t = -kappa_REV * r_{t-1}
            #
            # A positive coefficient creates demand against the
            # immediately preceding return. The default coefficient is
            # zero, preserving the canonical ABM dynamics.
            # ------------------------------------------------------------
            reversal_demand[t] = (
                -self.short_horizon_reversal_strength
                * previous_return
            )

            # ------------------------------------------------------------
            # 3c. Optional sparse symmetric rare-shock demand
            #
            # I_t ~ Bernoulli(p_shock)
            # S_t ~ Rademacher(-1, +1)
            # A_t = scale * exp(sigma Z_t - 0.5 sigma^2)
            # D_SHOCK,t = I_t * S_t * A_t
            #
            # A dedicated RNG stream prevents shock draws from consuming
            # random numbers from the canonical simulator stream.
            # ------------------------------------------------------------
            if (
                self.rare_shock_probability > 0.0
                and self.rare_shock_scale > 0.0
            ):
                event = (
                    self.rare_shock_rng.random()
                    < self.rare_shock_probability
                )
                rare_shock_event[t] = float(event)

                if event:
                    rare_shock_sign[t] = (
                        -1.0
                        if self.rare_shock_rng.random() < 0.5
                        else 1.0
                    )
                    lognormal_draw = self.rare_shock_rng.normal(
                        loc=0.0,
                        scale=1.0,
                    )
                    rare_shock_magnitude[t] = (
                        self.rare_shock_scale
                        * np.exp(
                            self.rare_shock_log_sigma * lognormal_draw
                            - 0.5 * self.rare_shock_log_sigma**2
                        )
                    )
                    rare_shock_demand[t] = (
                        rare_shock_sign[t]
                        * rare_shock_magnitude[t]
                    )

            # ------------------------------------------------------------
            # 4. Aggregate demand
            #
            # Formula:
            # D_t = (1 - x_t)D_F,t + x_t D_C,t + D_N,t
            #       + D_REV,t + D_SHOCK,t
            # ------------------------------------------------------------
            aggregate_demand[t] = (
                (1.0 - previous_chartist_share) * fundamentalist_demand[t]
                + previous_chartist_share * chartist_demand[t]
                + noise_demand[t]
                + reversal_demand[t]
                + rare_shock_demand[t]
            )

            # ------------------------------------------------------------
            # 5. Persistent market stress
            #
            # Demand stress:
            # large order imbalances increase market stress.
            #
            # Return stress:
            # large previous returns increase market stress.
            #
            # Then:
            # S_t = rho S_{t-1} + (1-rho) S_tilde
            # ------------------------------------------------------------
            demand_stress = abs(aggregate_demand[t]) / self.stress_demand_scale
            return_stress = abs(previous_return) / self.stress_return_scale

            stress_target[t] = 0.5 * demand_stress + 0.5 * return_stress

            stress_state[t] = (
                self.liquidity_persistence * previous_stress
                + (1.0 - self.liquidity_persistence) * stress_target[t]
            )

            stress_state[t] = np.clip(
                stress_state[t],
                0.0,
                self.max_stress
            )

            # ------------------------------------------------------------
            # 6. Endogenous liquidity with stress amplification
            #
            # L_t = L0 * exp(-gamma * |D_t| * (1 + a S_t))
            #
            # Interpretation:
            # when market stress is high, the same aggregate demand causes
            # a stronger deterioration in liquidity.
            # ------------------------------------------------------------
            stress_amplifier = 1.0 + (
                self.stress_impact_on_liquidity * stress_state[t]
            )

            raw_liquidity[t] = self.L0 * np.exp(
                -self.gamma * abs(aggregate_demand[t]) * stress_amplifier
            )

            liquidity[t] = max(
                self.liquidity_floor,
                raw_liquidity[t]
            )

            if self.enable_liquidity_provider:
                lp_state = self._compute_liquidity_provider_state(
                    baseline_liquidity=liquidity[t],
                    aggregate_demand=aggregate_demand[t],
                    previous_return=previous_return,
                    stress_state=stress_state[t],
                    previous_inventory=lp_inventory[t - 1],
                )
                lp_inventory[t] = lp_state["inventory"]
                lp_inventory_pressure[t] = lp_state["inventory_pressure"]
                lp_depth[t] = lp_state["depth"]
                lp_spread[t] = lp_state["spread"]
                lp_effective_liquidity[t] = lp_state["effective_liquidity"]
                lp_market_impact_multiplier[t] = lp_state[
                    "market_impact_multiplier"
                ]
                lp_liquidity_supply_multiplier[t] = lp_state[
                    "liquidity_supply_multiplier"
                ]
                lp_withdrawal_pressure[t] = lp_state[
                    "withdrawal_pressure"
                ]
                lp_effective_transaction_cost_proxy[t] = lp_state[
                    "transaction_cost_proxy"
                ]
                liquidity[t] = lp_effective_liquidity[t]
            else:
                lp_effective_liquidity[t] = liquidity[t]

            # ------------------------------------------------------------
            # 7. Price impact and log-return
            #
            # Formula:
            # r_t = (lambda / L_t) * D_t
            # ------------------------------------------------------------
            log_return[t] = (
                self.lambda_price / liquidity[t]
            ) * aggregate_demand[t] * lp_market_impact_multiplier[t]

            # Numerical safety:
            # prevents unrealistic numerical explosions.
            log_return[t] = np.clip(
                log_return[t],
                -self.max_abs_return,
                self.max_abs_return
            )

            # ------------------------------------------------------------
            # 8. Price update
            #
            # Formula:
            # P_t = P_{t-1} * exp(r_t)
            # ------------------------------------------------------------
            price[t] = previous_price * np.exp(log_return[t])

            # ------------------------------------------------------------
            # 9. Switching probabilities
            #
            # Baseline:
            # p_{F -> C,t} = eta + h x_t + beta max(r_{t-1}, 0)
            # p_{C -> F,t} = eta + h(1 - x_t) + beta max(-r_{t-1}, 0)
            #
            # Herding-biased extension:
            # p_{F -> C,t} receives + herding_bias
            # p_{C -> F,t} receives - herding_bias
            #
            # If herding_bias > 0, agents are more likely to move toward
            # chartist behavior and less likely to leave it.
            # ------------------------------------------------------------
            p_F_to_C[t], p_C_to_F[t] = self._compute_switching_probabilities(
                previous_chartist_share=previous_chartist_share,
                previous_return=previous_return,
            )

            # ------------------------------------------------------------
            # 10. Chartist share update
            #
            # x_{t+1} = x_t
            #           + (1 - x_t)p_{F -> C,t}
            #           - x_t p_{C -> F,t}
            # ------------------------------------------------------------
            inflow_to_chartists = (
                (1.0 - previous_chartist_share) * p_F_to_C[t]
            )

            outflow_from_chartists = (
                previous_chartist_share * p_C_to_F[t]
            )

            chartist_share[t] = (
                previous_chartist_share
                + inflow_to_chartists
                - outflow_from_chartists
            )

            chartist_share[t] = np.clip(chartist_share[t], 0.0, 1.0)

        results = pd.DataFrame(
            {
                "time": np.arange(self.T + 1),
                "price": price,
                "log_return": log_return,
                "chartist_share": chartist_share,
                "fundamentalist_share": 1.0 - chartist_share,
                "fundamentalist_demand": fundamentalist_demand,
                "chartist_demand": chartist_demand,
                "noise_demand": noise_demand,
                "reversal_demand": reversal_demand,
                "rare_shock_event": rare_shock_event,
                "rare_shock_sign": rare_shock_sign,
                "rare_shock_magnitude": rare_shock_magnitude,
                "rare_shock_demand": rare_shock_demand,
                "aggregate_demand": aggregate_demand,
                "raw_liquidity": raw_liquidity,
                "liquidity": liquidity,
                "lp_inventory": lp_inventory,
                "lp_inventory_pressure": lp_inventory_pressure,
                "lp_depth": lp_depth,
                "lp_spread": lp_spread,
                "lp_effective_liquidity": lp_effective_liquidity,
                "lp_market_impact_multiplier": lp_market_impact_multiplier,
                "lp_liquidity_supply_multiplier": (
                    lp_liquidity_supply_multiplier
                ),
                "lp_withdrawal_pressure": lp_withdrawal_pressure,
                "lp_effective_transaction_cost_proxy": (
                    lp_effective_transaction_cost_proxy
                ),
                "stress_state": stress_state,
                "stress_target": stress_target,
                "noise_scale": noise_scale,
                "noise_multiplier": noise_multiplier,
                "p_F_to_C": p_F_to_C,
                "p_C_to_F": p_C_to_F,
            }
        )

        return results
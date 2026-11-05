import numpy as np
from scipy.stats import norm


def compute_d1(S, K, r, sigma, tau):
    """
    Compute the Black-Scholes d1 term.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualized volatility.
    tau : float
        Time to maturity in years.

    Returns
    -------
    float
        Black-Scholes d1.
    """

    if S <= 0:
        raise ValueError("S must be positive.")

    if K <= 0:
        raise ValueError("K must be positive.")

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    if tau <= 0:
        raise ValueError("tau must be positive.")

    numerator = np.log(S / K) + (r + 0.5 * sigma**2) * tau
    denominator = sigma * np.sqrt(tau)

    d1 = numerator / denominator

    return d1


def compute_d2(S, K, r, sigma, tau):
    """
    Compute the Black-Scholes d2 term.

    Formula:
        d2 = d1 - sigma * sqrt(tau)
    """

    d1 = compute_d1(S=S, K=K, r=r, sigma=sigma, tau=tau)

    d2 = d1 - sigma * np.sqrt(tau)

    return d2


def european_call_price(S, K, r, sigma, tau):
    """
    Compute the Black-Scholes price of a European call option.

    Formula:
        C = S * N(d1) - K * exp(-r*tau) * N(d2)
    """

    if tau <= 0:
        return max(S - K, 0.0)

    d1 = compute_d1(S=S, K=K, r=r, sigma=sigma, tau=tau)
    d2 = compute_d2(S=S, K=K, r=r, sigma=sigma, tau=tau)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    return call_price


def european_put_price(S, K, r, sigma, tau):
    """
    Compute the Black-Scholes price of a European put option.

    Formula:
        P = K * exp(-r*tau) * N(-d2) - S * N(-d1)
    """

    if tau <= 0:
        return max(K - S, 0.0)

    d1 = compute_d1(S=S, K=K, r=r, sigma=sigma, tau=tau)
    d2 = compute_d2(S=S, K=K, r=r, sigma=sigma, tau=tau)

    put_price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return put_price


def call_delta(S, K, r, sigma, tau):
    """
    Compute the Black-Scholes delta of a European call option.

    Formula:
        Delta_call = N(d1)
    """

    if tau <= 0:
        if S > K:
            return 1.0
        return 0.0

    d1 = compute_d1(S=S, K=K, r=r, sigma=sigma, tau=tau)

    delta = norm.cdf(d1)

    return delta


def put_delta(S, K, r, sigma, tau):
    """
    Compute the Black-Scholes delta of a European put option.

    Formula:
        Delta_put = N(d1) - 1
    """

    if tau <= 0:
        if S < K:
            return -1.0
        return 0.0

    d1 = compute_d1(S=S, K=K, r=r, sigma=sigma, tau=tau)

    delta = norm.cdf(d1) - 1.0

    return delta
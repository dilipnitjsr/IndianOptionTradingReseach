"""Black-Scholes pricing for European index options and the transaction-cost model."""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

SQRT2 = sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / SQRT2))


def bs_price(spot: float, strike: float, tau: float, sigma: float, r: float, right: int, q: float = 0.0) -> float:
    """Black-Scholes price. ``right`` is +1 for a call and -1 for a put."""
    if tau <= 0.0:
        return max(right * (spot - strike), 0.0)
    vol = sigma * sqrt(tau)
    d1 = (log(spot / strike) + (r - q + 0.5 * sigma * sigma) * tau) / vol
    d2 = d1 - vol
    if right > 0:
        return spot * exp(-q * tau) * norm_cdf(d1) - strike * exp(-r * tau) * norm_cdf(d2)
    return strike * exp(-r * tau) * norm_cdf(-d2) - spot * exp(-q * tau) * norm_cdf(-d1)


def bs_delta(spot: float, strike: float, tau: float, sigma: float, r: float, right: int, q: float = 0.0) -> float:
    vol = sigma * sqrt(tau)
    d1 = (log(spot / strike) + (r - q + 0.5 * sigma * sigma) * tau) / vol
    return exp(-q * tau) * norm_cdf(d1) if right > 0 else -exp(-q * tau) * norm_cdf(-d1)


def select_strike(spot: float, tau: float, sigma: float, r: float, right: int,
                  target_delta: float = 0.5, step: float = 50.0, span: int = 8) -> float:
    """Listed strike whose absolute delta is closest to the target (strike-selection rule of the paper)."""
    centre = round(spot / step) * step
    best, best_gap = centre, 1e9
    for k in range(-span, span + 1):
        strike = centre + k * step
        gap = abs(abs(bs_delta(spot, strike, tau, sigma, r, right)) - target_delta)
        if gap < best_gap:
            best, best_gap = strike, gap
    return best


@dataclass(frozen=True)
class CostParams:
    """Charges on an option round trip (cost model of the paper). The rates are assumptions to be
    checked against the exchange and tax schedules in force for the test period."""

    brokerage: float = 20.0          # rupees per order
    stt: float = 0.0015              # on sell value
    exchange: float = 0.0003503      # on buy and sell value
    sebi: float = 0.000001           # on buy and sell value
    stamp: float = 0.00003           # on buy value
    gst: float = 0.18                # on brokerage, exchange and SEBI fees
    half_spread: float = 0.5         # rupees per unit, each side


def round_trip_cost(buy_value: float, sell_value: float, units: float, c: CostParams) -> float:
    turnover = buy_value + sell_value
    fees = (c.exchange + c.sebi) * turnover
    return (2.0 * c.brokerage + c.stt * sell_value + fees + c.stamp * buy_value
            + c.gst * (2.0 * c.brokerage + fees) + units * 2.0 * c.half_spread)

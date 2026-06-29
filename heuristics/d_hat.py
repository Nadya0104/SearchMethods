"""
Inadmissible distance-to-go estimator (d̂) for EES.

In EES, two heuristics serve different roles:
  h      — ADMISSIBLE cost-to-go  → used only to guarantee the w-bound (cleanup list)
  d̂      — INADMISSIBLE distance-to-go estimate → used to prefer nodes closer to a goal

For the standard 15-puzzle (unit costs), cost == distance, so d̂ = h is
a valid starting point and is what the EES paper uses for the tiles domain.

However, we can produce a stronger inadmissible d̂ using online single-step
error correction — the technique described in Thayer & Ruml (2011):

    Let  ε_d  = mean one-step error in d observed along the current path.
    Then  d̂(n) = d(n) / (1 - ε_d)   [when ε_d < 1]

Since computing per-node path statistics requires passing search-path
context into the heuristic (complicating the interface), we expose two
simpler strategies here that can be used without modifying the algorithm:

Strategy 1 — INFLATED:
    d̂(n) = d(n) * inflation_factor      (static, inadmissible for factor > 1)
    Simple and effective; essentially turns the distance estimate into a
    weighted version.  inflation_factor=1 recovers the admissible case.

Strategy 2 — COMBINED:
    d̂(n) = α * d(n) + (1-α) * h(n)     (convex combination)
    Useful when you have two heuristics of different strengths and want
    a smooth blend.

Both return callables compatible with the `d_hat` parameter of ees().

For Phase 1 (Manhattan Distance only), passing d_hat=manhattan_distance
directly to EES is perfectly correct and matches what the paper does for
unit-cost domains.  These wrappers become more useful in Phase 2+ when
you have PDB available as a stronger base estimator.
"""

from __future__ import annotations
from typing import Callable

H_Fn = Callable[[tuple[int, ...]], int | float]


def inflated_d_hat(
    d: H_Fn,
    factor: float = 1.2,
) -> H_Fn:
    """
    Return an inadmissible d̂ that inflates *d* by *factor*.

    d̂(n) = d(n) * factor

    Parameters
    ----------
    d      : base distance-to-go estimate (admissible or not)
    factor : inflation factor > 1.0 makes it inadmissible
    """
    def _d_hat(state: tuple[int, ...]) -> float:
        return d(state) * factor
    _d_hat.__name__ = f"inflated_d_hat(factor={factor})"
    return _d_hat


def combined_d_hat(
    d: H_Fn,
    h: H_Fn,
    alpha: float = 0.5,
) -> H_Fn:
    """
    Return a d̂ that blends distance and cost estimates.

    d̂(n) = alpha * d(n) + (1 - alpha) * h(n)

    alpha=1.0  →  pure distance estimate
    alpha=0.0  →  pure cost estimate (degenerates to skeptical search ordering)
    """
    def _d_hat(state: tuple[int, ...]) -> float:
        return alpha * d(state) + (1.0 - alpha) * h(state)
    _d_hat.__name__ = f"combined_d_hat(alpha={alpha})"
    return _d_hat


def identity_d_hat(h: H_Fn) -> H_Fn:
    """
    d̂ = h  (distance == cost in unit-cost domains).
    This is the default used by EES in the paper for the 15-puzzle.
    Wraps h with a descriptive name for logging purposes.
    """
    def _d_hat(state: tuple[int, ...]) -> int | float:
        return h(state)
    _d_hat.__name__ = f"identity_d_hat({getattr(h, '__name__', repr(h))})"
    return _d_hat
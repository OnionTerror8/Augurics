"""Retail / wholesale / manufacturing feature profiles. Task 2.2.

A profile is just a named, declared subset + parameterisation of the v1
feature library -- declared here so choosing one is a config decision,
not an ad hoc one made per model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureProfile:
    name: str
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    adstock_decay: float
    use_price_features: bool = True


RETAIL_PROMO_V1 = FeatureProfile(
    name="retail_promo_v1",
    lags=(7, 28),
    rolling_windows=(7, 28),
    adstock_decay=0.5,
    use_price_features=True,
)

WHOLESALE_V1 = FeatureProfile(
    name="wholesale_v1",
    lags=(7, 14, 28),
    rolling_windows=(14, 28),
    adstock_decay=0.3,
    use_price_features=True,
)

MANUFACTURING_V1 = FeatureProfile(
    name="manufacturing_v1",
    lags=(28,),
    rolling_windows=(28, 90),
    adstock_decay=0.2,
    use_price_features=False,
)

PROFILES: dict[str, FeatureProfile] = {
    p.name: p for p in (RETAIL_PROMO_V1, WHOLESALE_V1, MANUFACTURING_V1)
}


def get_profile(name: str) -> FeatureProfile:
    if name not in PROFILES:
        raise KeyError(f"Unknown feature profile '{name}'. Known profiles: {sorted(PROFILES)}")
    return PROFILES[name]

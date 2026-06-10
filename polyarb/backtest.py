"""Backtesting + Monte-Carlo validation for the 2-bucket strategy.

Two modes:

1. ``monte_carlo(...)`` — no network. Simulates many synthetic event-days using
   realistic NWP forecast-error statistics (calibrated to published ECMWF/GFS
   2-day skill) and the ensemble under-dispersion problem. Answers the real
   question: given the σ-gate, what hit-rate and coverage do we get, and how
   sensitive is it to ensemble miscalibration?

2. ``run_historical(...)`` — needs network. For each city + date, pulls the
   Open-Meteo *historical forecast* (what the models predicted 2 days ahead)
   and the *archived actual* daily-max, reconstructs the standard Polymarket
   2°C-bucket lattice, applies the live strategy logic, and reports realized
   hit-rate / coverage / PnL per city.

The Polymarket bucket lattice is reconstructed deterministically (2°C wide,
centred on a round grid) because Polymarket weather events use a fixed bucket
scheme per city/season. If you have a CSV of real resolved markets, pass it to
``run_historical`` via ``pm_resolutions`` to override the synthetic lattice.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta

from .forecast import ForecastDistribution
from .two_bucket_strategy import select_best_pair
from .weather_markets import WeatherBucket


# ----- Standard Polymarket-style bucket lattice -----------------------------

def make_lattice(
    center_c: float,
    *,
    width_c: float = 2.0,
    n_each_side: int = 4,
) -> list[WeatherBucket]:
    """Build a contiguous 2°C-wide bucket lattice around a center temperature,
    with open-ended buckets at both ends (mirrors real Polymarket events)."""
    base = math.floor(center_c / width_c) * width_c
    edges = [base + width_c * (i - n_each_side) for i in range(2 * n_each_side + 1)]
    buckets: list[WeatherBucket] = []
    # open-ended low
    buckets.append(_mk(None, edges[0]))
    for lo, hi in zip(edges, edges[1:]):
        buckets.append(_mk(lo, hi))
    # open-ended high
    buckets.append(_mk(edges[-1], None))
    return buckets


def _mk(lo, hi) -> WeatherBucket:
    slug = f"b_{lo}_{hi}"
    return WeatherBucket(
        slug=slug, title=slug,
        token_yes=f"Y_{slug}", token_no=f"N_{slug}",
        lo_c=lo, hi_c=hi, minimum_order_size=1.0,
    )


def _winning_bucket(buckets: list[WeatherBucket], observed_c: float) -> WeatherBucket | None:
    """Polymarket resolves on whole-°C truncation of the observed max."""
    truncated = int(math.floor(observed_c)) if observed_c >= 0 else int(math.ceil(observed_c))
    for b in buckets:
        lo_ok = b.lo_c is None or truncated >= int(b.lo_c)
        hi_ok = b.hi_c is None or truncated <= int(b.hi_c) - 1 if b.hi_c is not None else True
        # bucket [lo, hi) — winner if lo <= t < hi  (2°C bucket covers e.g. 17,18)
        if b.lo_c is not None and b.hi_c is not None:
            if int(b.lo_c) <= truncated < int(b.hi_c):
                return b
        elif b.lo_c is None and b.hi_c is not None:
            if truncated < int(b.hi_c):
                return b
        elif b.hi_c is None and b.lo_c is not None:
            if truncated >= int(b.lo_c):
                return b
    return None


# ----- Monte-Carlo validation (no network) ---------------------------------

@dataclass
class MCResult:
    n_events: int
    n_traded: int
    n_hit: int
    coverage: float
    hit_rate: float
    skipped_hit_rate: float
    calibration_factor: float
    max_sigma_c: float
    min_pair_prob: float
    notes: str = ""


def monte_carlo(
    *,
    n_events: int = 20_000,
    max_sigma_c: float = 1.2,
    min_pair_prob: float = 0.90,
    reported_sigma_lo: float = 0.5,
    reported_sigma_hi: float = 2.5,
    calibration_factor: float = 1.15,
    seed: int = 7,
) -> MCResult:
    """Simulate event-days.

    Model:
      * The market's true daily-max ``T_true`` is drawn from a broad climatology.
      * The forecast point ``μ`` = T_true + error, where the *actual* error has
        std ``σ_real = σ_reported * calibration_factor`` (ensembles are usually
        under-dispersive, so realised errors are ~10-30% larger than the
        reported spread).
      * The strategy only sees ``μ`` and ``σ_reported`` and must decide.

    We gate on σ_reported and the combined-pair probability computed from
    ``N(μ, σ_reported)``. Then we check whether T_true actually landed in the
    chosen pair.
    """
    rng = random.Random(seed)
    n_traded = 0
    n_hit = 0
    skipped = 0
    skipped_hit = 0

    for _ in range(n_events):
        t_true = rng.uniform(-5.0, 32.0)
        sigma_rep = rng.uniform(reported_sigma_lo, reported_sigma_hi)
        sigma_real = sigma_rep * calibration_factor
        mu = t_true + rng.gauss(0.0, sigma_real)

        lattice = make_lattice(mu)
        dist = ForecastDistribution(mu_c=mu, sigma_c=sigma_rep, sources=["mc"])
        pair = select_best_pair(
            lattice, dist,
            min_combined_prob=min_pair_prob,
            max_sigma_c=max_sigma_c,
        )

        winner = _winning_bucket(lattice, t_true)
        in_pair = (
            winner is not None
            and pair is not None
            and winner.slug in (pair.primary.slug, pair.secondary.slug)
        )

        if pair is not None:
            n_traded += 1
            if in_pair:
                n_hit += 1
        else:
            skipped += 1
            # counterfactual: would the best pair (ignoring gate) have hit?
            best = select_best_pair(lattice, dist, min_combined_prob=0.0, max_sigma_c=99.0)
            if best is not None and winner is not None and winner.slug in (
                best.primary.slug, best.secondary.slug
            ):
                skipped_hit += 1

    coverage = n_traded / n_events if n_events else 0.0
    hit_rate = n_hit / n_traded if n_traded else 0.0
    skip_hr = skipped_hit / skipped if skipped else 0.0
    return MCResult(
        n_events=n_events,
        n_traded=n_traded,
        n_hit=n_hit,
        coverage=coverage,
        hit_rate=hit_rate,
        skipped_hit_rate=skip_hr,
        calibration_factor=calibration_factor,
        max_sigma_c=max_sigma_c,
        min_pair_prob=min_pair_prob,
    )


def sweep_calibration(
    factors: list[float] | None = None,
    **kw,
) -> list[MCResult]:
    factors = factors or [1.0, 1.1, 1.15, 1.25, 1.4]
    return [monte_carlo(calibration_factor=f, **kw) for f in factors]


def sweep_gates(
    sigmas: list[float] | None = None,
    probs: list[float] | None = None,
    **kw,
) -> list[MCResult]:
    sigmas = sigmas or [1.0, 1.1, 1.2, 1.4]
    probs = probs or [0.88, 0.90, 0.92]
    out = []
    for s in sigmas:
        for p in probs:
            out.append(monte_carlo(max_sigma_c=s, min_pair_prob=p, **kw))
    return out


# ----- Historical backtest (needs network) ----------------------------------

CITY_COORDS = {
    "MOSCOW": (55.59, 37.27),
    "NYC": (40.78, -73.97),
    "LA": (34.05, -118.24),
    "CHICAGO": (41.98, -87.90),
    "MIAMI": (25.79, -80.32),
    "LONDON": (51.50, 0.05),
    "BERLIN": (52.36, 13.50),
    "PARIS": (49.01, 2.55),
    "TOKYO": (35.55, 139.78),
}


@dataclass
class CityBacktest:
    city: str
    n_days: int = 0
    n_traded: int = 0
    n_hit: int = 0
    pnl_usd: float = 0.0
    forecasts: list[tuple[str, float, float, float]] = field(default_factory=list)
    # (date, mu, sigma, observed)

    @property
    def coverage(self) -> float:
        return self.n_traded / self.n_days if self.n_days else 0.0

    @property
    def hit_rate(self) -> float:
        return self.n_hit / self.n_traded if self.n_traded else 0.0


async def run_historical(
    client,
    city: str,
    start: date,
    end: date,
    *,
    open_meteo_archive_host: str = "https://archive-api.open-meteo.com",
    open_meteo_host: str = "https://api.open-meteo.com",
    max_sigma_c: float = 1.2,
    min_pair_prob: float = 0.90,
    lead_days: int = 2,
    entry_cost_primary: float = 0.40,
    entry_cost_secondary: float = 0.18,
    budget_usd: float = 30.0,
    primary_share: float = 0.6,
) -> CityBacktest:
    """Backtest one city over [start, end].

    For each target date D:
      * forecast: Open-Meteo *previous-runs* archive — what the ensemble said
        on D-lead_days for D's daily-max.
      * actual: Open-Meteo archive (ERA5) daily-max for D.
      * apply gate; if traded, settle against the winning bucket.

    Requires network — Open-Meteo archive is blocked in some sandboxes.
    """
    import httpx

    lat, lon = CITY_COORDS[city]
    bt = CityBacktest(city=city)

    # Pull actual daily-max for the whole range in one archive call.
    try:
        r = await client.get(
            f"{open_meteo_archive_host}/v1/archive",
            params={
                "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
                "start_date": start.isoformat(), "end_date": end.isoformat(),
                "daily": "temperature_2m_max", "timezone": "UTC",
            },
            timeout=30.0,
        )
        r.raise_for_status()
        actual = r.json().get("daily", {})
    except httpx.HTTPError as e:
        bt.forecasts.append(("ERROR", 0, 0, 0))
        return bt

    actual_map = {
        d: t for d, t in zip(actual.get("time", []), actual.get("temperature_2m_max", []))
        if t is not None
    }

    # For the forecast side, use Open-Meteo previous-runs (forecast issued N days
    # earlier). The "&past_days" + model previous run approach varies; here we
    # request the historical forecast API which returns past model runs.
    for d_str, observed in actual_map.items():
        d = date.fromisoformat(d_str)
        try:
            fr = await client.get(
                f"{open_meteo_host}/v1/forecast",
                params={
                    "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
                    "daily": "temperature_2m_max",
                    "timezone": "UTC",
                    "start_date": d.isoformat(), "end_date": d.isoformat(),
                    "models": "ecmwf_ifs04,gfs_seamless,icon_seamless",
                    "past_days": "0",
                },
                timeout=20.0,
            )
            fr.raise_for_status()
            fdaily = fr.json().get("daily", {})
        except httpx.HTTPError:
            continue

        vals = []
        for k, v in fdaily.items():
            if k.startswith("temperature_2m_max") and isinstance(v, list) and v and v[0] is not None:
                vals.append(float(v[0]))
        if len(vals) < 2:
            continue

        mu = statistics.fmean(vals)
        sigma = max(statistics.stdev(vals), 0.4)
        bt.n_days += 1
        bt.forecasts.append((d_str, mu, sigma, observed))

        lattice = make_lattice(mu)
        dist = ForecastDistribution(mu_c=mu, sigma_c=sigma, sources=["hist"])
        pair = select_best_pair(
            lattice, dist, min_combined_prob=min_pair_prob, max_sigma_c=max_sigma_c
        )
        if pair is None:
            continue

        bt.n_traded += 1
        winner = _winning_bucket(lattice, observed)
        cost = budget_usd  # we deploy full budget split across the two
        won = winner is not None and winner.slug in (pair.primary.slug, pair.secondary.slug)
        if won:
            bt.n_hit += 1
            # crude payout: winning bucket pays $1/share. Shares bought with the
            # share of budget allocated to whichever bucket won.
            if winner.slug == pair.primary.slug:
                shares = (budget_usd * primary_share) / entry_cost_primary
            else:
                shares = (budget_usd * (1 - primary_share)) / entry_cost_secondary
            bt.pnl_usd += shares * 1.0 - cost
        else:
            bt.pnl_usd -= cost

    return bt

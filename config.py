"""Central configuration. Everything tunable lives here, nothing tunable lives elsewhere."""
from pathlib import Path
import os

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRICES = DATA / "prices"
UNIVERSE = DATA / "universe"
SIGNALS = DATA / "signals"
REPORTS = ROOT / "reports"

# Synthetic-data sandbox for `demo`. Kept entirely separate from the real
# directories above so demo runs can never overwrite real prices, universe
# snapshots, or the signal log (which is timestamped evidence, not cache).
DEMO_PRICES = DATA / "demo_prices"
DEMO_UNIVERSE = DATA / "demo_universe"

# Local cache of ticker -> SIC code, sourced from SEC (Polygon's bulk
# reference endpoint doesn't return SIC — see sic.py). Rebuildable, not
# evidence, so it lives outside data/signals/ and is gitignored like prices.
REFERENCE = DATA / "reference"
SIC_MAP = REFERENCE / "sic_map.parquet"

for d in (PRICES, UNIVERSE, SIGNALS, REPORTS, DEMO_PRICES, DEMO_UNIVERSE, REFERENCE):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- SEC (SIC lookup)
# SEC requires a descriptive User-Agent identifying the requester on every
# request to sec.gov / data.sec.gov, or it may block the client.
SEC_USER_AGENT = "stock-scanner (messaoudi.lokman@gmail.com)"
SEC_REQUESTS_PER_SEC = 8          # SEC's stated guidance is ~10/sec; stay under it
SIC_REFRESH_DAYS = 30             # SIC codes almost never change

# ---------------------------------------------------------------- data source
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
POLYGON_BASE = "https://api.polygon.io"
REQUEST_SLEEP = 12.0          # paid tier: unlimited. free tier: set to 12.0

# ---------------------------------------------------------------- universe
MIN_PRICE = 5.0
MIN_ADV_20 = 5_000_000        # 20-day average dollar volume
MIN_MARKET_CAP = 500_000_000  # applied only where share counts are available
ALLOWED_TYPES = {"CS"}        # common stock only. excludes ETF/ETN/FUND/WARRANT/UNIT/RIGHT
ALLOWED_EXCHANGES = {"XNYS", "XNAS", "XASE"}  # NYSE, Nasdaq, NYSE American

# ---------------------------------------------------------------- features
LOOKBACK = 60                 # rolling window for location/scale estimates
MIN_OBS = 40                  # minimum observations before a z-score is trusted
BREAKOUT_WINDOWS = (20, 60, 252)
HISTORY_DAYS = 520            # ~2 years of trading days to retain

# Robust scale: MAD * 1.4826 is a consistent estimator of sigma for normal data,
# and unlike the sample std it is not inflated by a handful of earnings gaps.
MAD_SCALE = 1.4826
ROBUST_SCALE = True

# Continuation features (deterministic columns only — see features.py). These
# are NOT inputs to score_day's weighted blend; the weights on any of them
# will be fitted from forward returns later, not asserted now.
GAP_RETENTION_MIN_GAP = 0.001   # min |open/prev_close - 1| before gap_retention is computed, else NaN
ATR_SHORT_WINDOW = 5            # pre-event compression numerator: ATR over this many PRIOR sessions
ATR_LONG_WINDOW = 60            # ...divided by ATR over this many prior sessions

# ---------------------------------------------------------------- thresholds
# Absolute gates. These are what let the scanner return FEWER than N names —
# or none at all — on a quiet day. Rank-based scoring alone cannot do that.
# Calibrated against 90 days of real data (September 2026). Do not retune
# until `evaluate` has at least six months of forward returns to judge against.
Z_RETURN_GATE = 3.5
Z_VOLUME_GATE = 3.0
Z_VOLUME_STRONG = 4.5
Z_RETURN_QUIET = 0.5          # "high participation, low resolution" needs a SMALL price move

MAX_PER_BUCKET = 10
WINSORISE_AT = 5.0            # applied to z-scores before display, not before ranking

# Sanity guard: a move this large with no volume confirmation is almost always
# a corporate action the adjustment pipeline missed, not information.
CORP_ACTION_RET_GUARD = 0.40
CORP_ACTION_VOL_FLOOR = 3.0

# ---------------------------------------------------------------- scoring
# NOTE: these weights are a stated ATTENTION allocation, not a fitted
# probability of a subsequent move. Do not let them become a predictive claim.
# Run `python main.py evaluate` once you have ~6 months of logged signals to
# see whether the composite predicts anything at all before trusting it.
# "raw"        — the five-feature blend as originally specified
# "orthogonal" — collapses the three highly-correlated price features into one
#                axis before blending. Run `diagnose` to see why this matters:
#                ret_z, rel_ret and gap_z correlate ~0.9+, so the raw weights
#                put 60% of the score on one signal and 30% on volume, which is
#                the only genuinely independent input (corr with price ~0.00).
SCORE_MODE = "orthogonal"

SCORE_WEIGHTS = {
    "volume_rank": 0.30,
    "return_rank": 0.25,
    "relative_rank": 0.20,
    "gap_rank": 0.15,
    "breakout_rank": 0.10,
}

# Two near-independent axes plus a weak third.
SCORE_WEIGHTS_ORTHOGONAL = {
    "price_rank": 0.45,      # max of return / relative / gap ranks — one event, one vote
    "volume_rank": 0.40,     # the orthogonal axis, and the harder one to fake
    "breakout_rank": 0.15,
}

# ---------------------------------------------------------------- calendar
# Known mechanical-flow days (index reconstitutions, options expiries) are
# tagged on the signal log via flow_calendar.py, not filtered — see that
# module for the explicit per-source rules.

FORWARD_HORIZONS = (1, 5, 20)  # trading days, for the evaluation harness

# Path statistics (MFE/MAE, event-time curves) in evaluate(). Separate from
# FORWARD_HORIZONS above: those are point-in-time snapshots, these describe
# the whole path from trigger to tau sessions later.
MFE_MAE_WINDOW = 20        # sessions after trigger over which MFE/MAE are taken
EVENT_TIME_MAX_TAU = 60    # sessions after trigger the event-time curve covers

# ---------------------------------------------------------------- email
# "Breadth" in the email subject line: today's raw gate crossings vs their
# own trailing 90-day median (same numbers as the report's "crossings:" line).
BREADTH_QUIET_RATIO = 0.5      # today <= this x median -> "quiet"
BREADTH_ELEVATED_RATIO = 1.5   # today <= this x median -> "normal", above -> "elevated"

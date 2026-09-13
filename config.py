"""Central configuration. Everything tunable lives here, nothing tunable lives elsewhere."""
from pathlib import Path
import os

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PRICES = DATA / "prices"
UNIVERSE = DATA / "universe"
SIGNALS = DATA / "signals"
REPORTS = ROOT / "reports"

for d in (PRICES, UNIVERSE, SIGNALS, REPORTS):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data source
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
POLYGON_BASE = "https://api.polygon.io"
REQUEST_SLEEP = 0.02          # paid tier: unlimited. free tier: set to 12.0

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

# ---------------------------------------------------------------- thresholds
# Absolute gates. These are what let the scanner return FEWER than N names —
# or none at all — on a quiet day. Rank-based scoring alone cannot do that.
Z_RETURN_GATE = 2.0
Z_VOLUME_GATE = 2.0
Z_VOLUME_STRONG = 3.0
Z_RETURN_QUIET = 1.0          # "silent accumulation" needs a SMALL price move

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
# Days on which index rebalances flood the tape with non-informational volume.
# Triple witching / quarterly rebalance: third Friday of Mar, Jun, Sep, Dec.
FLAG_REBALANCE_DAYS = True

FORWARD_HORIZONS = (1, 5, 20)  # trading days, for the evaluation harness

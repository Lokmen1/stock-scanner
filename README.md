# Anomaly Scanner — V1

Daily statistical scan of the US equity market. No LLM, no news, no options.
Input: daily OHLCV. Output: a short list of names that did something unusual,
with enough context to start asking *why*.

## Run it now, without an API key

```bash
pip install -r requirements.txt
python main.py demo
```

Generates a synthetic 600-name market with six known anomalies planted on the
final session plus one fake corporate action, then runs the full pipeline. If
the scanner finds the six and suppresses the seventh, the wiring is correct.

## Run it on real data

```bash
export POLYGON_API_KEY=...
python main.py universe          # snapshot point-in-time universe membership
python main.py backfill --days 520
python main.py scan              # nightly
```

Then nightly at ~22:00 London:

```cron
0 22 * * 1-5 cd /path/to/stock_scanner && python main.py universe && python main.py scan
```

## Commands

| | |
|---|---|
| `demo` | synthetic data + full run, no API key needed |
| `backfill --days N` | pull N sessions of grouped daily aggregates |
| `universe` | snapshot today's universe membership (run daily) |
| `scan` | compute features, rank, print + log |
| `diagnose` | feature correlations and gate hit-rates |
| `evaluate` | forward returns on logged signals |

## Design decisions that differ from the original spec

**Robust z-scores (median/MAD) rather than mean/std.** A stock that gaps 9%
every quarter carries an inflated 60-day sigma for the following three months,
so its next genuine surprise gets scored *down* precisely because it has a
history of surprises. Median and MAD are insensitive to a handful of jump days.
Set `ROBUST_SCALE = False` in config to compare.

**Absolute gates, not just top-N.** A pure ranking emits N names every evening
whether or not anything happened. With absolute thresholds a quiet day returns
a short list, or none. In the demo ~17% of sessions produce nothing, which is
roughly the right band — 0% means the gates are decorative, above ~40% means
no research flow.

Real data does not behave like the demo here. Against 90 days of real Polygon
data (September 2026), the demo-tuned gates (`Z_RETURN_GATE = 2.0`, etc.) never
produced a no-flag day — a ~2,650-name universe throws that many genuine
single-name earnings/news days that a 2σ own-history move is common, not rare.
Gates were re-calibrated to `Z_RETURN_GATE = 3.5`, `Z_VOLUME_GATE = 3.0`,
`Z_VOLUME_STRONG = 4.5`, `Z_RETURN_QUIET = 0.5` (see the comment in
`config.py`). At those levels, only a fairly extreme single-name event clears
the bar — roughly a 3.5σ own-history return move on 3σ volume. Do not retune
further until `evaluate` has at least six months of forward returns to judge
against.

**Orthogonalised scoring.** Run `diagnose`: `ret_z`, `rel_ret` and `gap_z`
correlate at ~0.9+. The original weights therefore put 0.25+0.20+0.15 = 60% of
the score on a single underlying signal and only 30% on volume, which is the
one genuinely independent input (correlation with price ≈ 0.00). Default mode
collapses the three price features into one axis. `SCORE_MODE = "raw"` restores
the original behaviour.

**Survivorship handling.** Grouped-daily bars are stored for every ticker that
traded, before any filter, so price history is survivorship-free. Universe
membership is snapshotted daily and never overwritten, building a genuine
point-in-time record going forward. What cannot be fixed: membership *before*
your first run. Backtests over the backfilled period stay contaminated — treat
forward-logged signals as the real evidence.

**Corporate action guard.** Polygon's `adjusted=true` handles splits and
dividends, but not everything. A >40% move on under-3σ volume is flagged and
suppressed: that combination is almost always an unhandled corporate action
rather than information.

**Signal logging from day one.** Every flag is written to `data/signals/` with
a timestamp. After ~6 months, `evaluate` joins these to subsequent prices and
reports forward returns against the market, per bucket and for the top-scored
quintile. If top-quintile excess return isn't clearly above whole-basket excess,
the scoring layer isn't earning its place and the weights are noise.

`data/signals/` was empty as of 13 Sept 2026 (an earlier demo run had leaked
synthetic entries into it; these were found and removed, and `log_signals` now
refuses to write anything matching the synthetic ticker pattern). The first
genuine entry dates from the first real `scan` after that.

## What the score is and is not

It is an **attention allocation** — how much research time a name deserves
tonight. It is not a probability and it is not fitted to anything. The weights
are asserted. Use `evaluate` to find out whether they predict anything before
letting them harden into a claim.

## Known limitations

- Market proxy is the cross-sectional median return, not a cap-weighted index.
  A SIC→sector map now exists (`sic.py`, `data/reference/sic_map.parquet`,
  ~98% coverage), used for clustering in `evaluate()`. Relative-strength
  scoring itself is still computed against the whole universe, not sector
  peers — using the map for that is the next upgrade.
- Market cap filter applies only where Polygon supplies share counts. Names
  with missing counts are kept rather than silently dropped.
- Holiday handling is crude — non-trading days return empty and are skipped.
- Short interest and borrow cost are not yet wired in. Known mechanical-flow
  days (S&P/FTSE Russell/Nasdaq-100/MSCI rebalances, options expiries) ARE
  covered — see `flow_calendar.py` — but these are not excluded from scoring;
  they're tagged via `mechanical_flow_possible` on the signal log instead,
  since forced flow is real supply/demand information, not noise to delete.

## Structural caveat

This finds things that have **already moved**. The original design had two
independent paths — anomaly detection *and* news/filings scanning, so catalysts
could surface before price reacts. This is only the first. The strategy it
implies is reaction, not anticipation. That's a reasonable place to start, since
it's the tractable half and it generates the raw material for a later
interpretation layer, but be clear that "here's something that happened, go find
out why" is a different business from "here's something mispriced".

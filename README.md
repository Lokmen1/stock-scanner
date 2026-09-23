# Anomaly Scanner — V1

Daily statistical scan of the US equity market. No LLM, no news, no options.
Input: daily OHLCV. Output: a short list of names that did something unusual,
with enough context to start asking *why*.

## How it works, end to end

- **Universe** (`universe`) — snapshot today's tradeable common-stock universe
  from Polygon (price/liquidity/market-cap filters applied later, not here).
  Never overwrites an earlier day's snapshot, so membership becomes a genuine
  point-in-time record going forward.
- **Backfill** (`backfill --days N`) — pull N sessions of grouped-daily OHLCV
  for *every* ticker that traded, before any filter, so price history stays
  survivorship-free even though today's universe list is not.
- **Features** (`features.py`) — for each name, robust (median/MAD) z-scores
  of return, volume, gap and relative-strength, plus 20/60/252-day breakout
  levels. Robust, not mean/std, so a name's own history of jump days doesn't
  mute its next genuine surprise.
- **Corporate-action guard** — a huge move on ordinary volume is flagged and
  suppressed rather than scored: that combination is almost always an
  unhandled split/spin-off, not information.
- **Gates → three buckets** (`rank.py`) — absolute thresholds (not top-N) sort
  survivors into `momentum`, `dislocation`, and `high_participation_low_resolution`
  (heavy volume, price barely moved — deliberately not called "accumulation",
  since the pattern is equally consistent with distribution, a block cross,
  or index flow). A quiet day can return a short list, or none.
- **Scoring** — an orthogonalised, percentile-rank blend across the surviving
  names: an *attention* ranking (how much research time a name deserves
  tonight), not a probability and not fitted to anything.
- **Episode tracking** (`episodes.py`) — the unit of analysis is an episode,
  not a ticker-day. A name's trigger extends its open episode if the previous
  one was within 5 trading sessions; a longer gap starts a new one. Tracks
  `event_age`, `days_triggered`, `max_score`, `first_trigger_date`.
- **Mechanical-flow tagging** (`flow_calendar.py`) — known index-rebalance and
  options-expiry days (S&P, FTSE Russell, Nasdaq-100, MSCI) are *tagged*
  (`mechanical_flow_possible`), never excluded: forced flow is real
  supply/demand information, just not fundamental information.
- **Sector mapping** (`sic.py`) — SEC-sourced SIC→sector lookup (Polygon's
  bulk endpoint doesn't provide it), cached locally, used to cluster
  observations for the evaluation harness's confidence intervals.
- **Signal log** (`data/signals/`) — every flag, permanently, from day one.
  This is evidence, not cache, and is committed to the repo.
- **Email** (`mailer.py`) — the nightly report is sent by SMTP (stdlib only)
  after the report file and signal log are already written, so a mail
  failure never costs the day's data. `--no-email` / `testmail` for testing.
- **Evaluate** (`evaluate`) — once signals accumulate, joins them to
  subsequent prices and reports forward returns at both the ticker-day and
  episode level, with cluster-bootstrapped confidence intervals (clustering
  by date × sector, so one shared shock across many names counts as one
  observation, not many), plus MFE/MAE and event-time curves showing whether
  a signal continues, fades, or reverses.

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
# .env: POLYGON_API_KEY=... (and, optionally, SMTP_* + REPORT_RECIPIENT for email)
python main.py universe          # snapshot point-in-time universe membership
python main.py backfill --days 520
python main.py sic-update        # optional, one-off/monthly: backfills the sector map
python main.py scan              # nightly
```

`run_scan.sh` wraps the `backfill --days 1` + `universe` + `scan` calls
(activating the venv and loading `.env` first) and is what's actually
scheduled — via `launchd` on macOS
(`~/Library/LaunchAgents/com.scanner.daily.plist`, Mon–Fri 07:00) or an
equivalent cron entry elsewhere:

```cron
0 7 * * 1-5 /path/to/stock_scanner/run_scan.sh
```

Runs in the **morning**, not right after the prior close. `backfill`'s
`--days N` always stops at "yesterday relative to when it runs" (`main.py`
`cmd_backfill`), because Polygon's grouped-daily aggregates for a session
aren't reliably finalized right after that session's own close. Run the job
that same evening and "yesterday" is one session behind the close you actually
want; run it the next morning and "yesterday" IS the most recent close, by
then fully finalized. The report is dated by session (`last`, not send time),
so a Wednesday-morning email is headed `Scan 2026-09-22` — Tuesday's close.

## Commands

| | |
|---|---|
| `demo` | synthetic data + full run, no API key needed |
| `backfill --days N` | pull N sessions of grouped daily aggregates |
| `universe` | snapshot today's universe membership (run daily) |
| `sic-update` | refresh the SEC SIC→sector cache (not on the scan path) |
| `scan` | compute features, rank, print, log, and email the report |
| `scan --no-email` | same, without sending the email (for testing) |
| `testmail` | send a single test email to verify SMTP config |
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

# Stock Scanner

Daily statistical anomaly scan of US equities. Python, pandas, Parquet.
No LLM layer yet. Deliberately small — do not add frameworks, databases,
schedulers or abstraction layers.

## Rules
- Always `source .venv/bin/activate` before running anything.
- NEVER commit .env or data/prices/. NEVER print or echo the API key.
- data/signals/ IS committed. It is timestamped evidence, not cache.
- All tunable parameters live in config.py. Nowhere else.
- Do NOT replace the robust median/MAD scale with mean/std. It is deliberate.
- Do NOT remove the absolute gates in rank.py in favour of plain top-N.
  The gates are what allow a quiet day to return nothing.
- The score is an attention ranking, not a probability. Never write comments
  or docs implying it predicts returns.
- Before any change to features.py or rank.py, run `python main.py demo` and
  confirm all six planted anomalies are found and S0007 is suppressed.

## Verify
python main.py demo       # six plants found, one corp action suppressed
python main.py diagnose   # correlations + gate hit rate

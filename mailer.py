"""Email delivery for the nightly report. Stdlib only (smtplib + email.message)
— no new dependency for something this small.

Config comes from the environment (SMTP_HOST, SMTP_PORT, SMTP_USER,
SMTP_PASSWORD, REPORT_RECIPIENT), loaded from .env by run_scan.sh before
main.py starts — same mechanism config.py already uses for POLYGON_API_KEY.
Defaults assume Gmail (smtp.gmail.com:587, STARTTLS). Gmail requires an App
Password for SMTP auth, not the account password.

SMTP_PASSWORD is never printed, logged, or embedded in an exception message
here — only the names of missing config vars are, which carries no secret.
"""
from __future__ import annotations

import html
import os
import smtplib
from email.message import EmailMessage

import config as C

DEFAULT_SMTP_HOST = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587

BUCKET_ABBR = {
    "momentum": "momentum",
    "dislocation": "dislocation",
    "high_participation_low_resolution": "HPLR",
}
BUCKET_ORDER = ("momentum", "dislocation", "high_participation_low_resolution")


def _config() -> dict:
    return {
        "host": os.environ.get("SMTP_HOST", DEFAULT_SMTP_HOST),
        "port": int(os.environ.get("SMTP_PORT", DEFAULT_SMTP_PORT)),
        "user": os.environ.get("SMTP_USER", ""),
        "password": os.environ.get("SMTP_PASSWORD", ""),
        "recipient": os.environ.get("REPORT_RECIPIENT", ""),
    }


def _require_config(cfg: dict) -> None:
    env_names = {"user": "SMTP_USER", "password": "SMTP_PASSWORD", "recipient": "REPORT_RECIPIENT"}
    missing = [env_names[k] for k in env_names if not cfg[k]]
    if missing:
        raise RuntimeError(f"Missing email config in .env: {', '.join(missing)}")


def _html_wrap(text: str) -> str:
    """The report is a monospace-aligned ASCII table — <pre> + a monospace
    font stack is what keeps that alignment intact in an HTML-rendering
    mail client."""
    return (
        "<html><body>"
        "<pre style=\"font-family: 'SF Mono', Menlo, Consolas, 'Courier New', "
        "monospace; font-size: 13px; line-height: 1.35; white-space: pre;\">"
        f"{html.escape(text)}"
        "</pre></body></html>"
    )


def _send(subject: str, text_body: str, html_body: str, cfg: dict) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["user"]
    msg["To"] = cfg["recipient"]
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as smtp:
        smtp.starttls()
        smtp.login(cfg["user"], cfg["password"])
        smtp.send_message(msg)


def _breadth_label(crossings_today: dict, crossings_median: dict) -> str:
    """Quiet/normal/elevated, from today's raw (pre-cap) gate crossings
    against their own trailing 90-day median — the same numbers the report
    header's 'crossings:' line already shows. Thresholds in config.py."""
    today_sum = sum(crossings_today.values())
    median_sum = sum(crossings_median.values())
    if median_sum <= 0:
        return "n/a"
    ratio = today_sum / median_sum
    if ratio <= C.BREADTH_QUIET_RATIO:
        return "quiet"
    if ratio <= C.BREADTH_ELEVATED_RATIO:
        return "normal"
    return "elevated"


def build_subject(date, buckets: dict, crossings_today: dict | None = None,
                   crossings_median: dict | None = None) -> str:
    counts = {k: len(v) for k, v in buckets.items()}
    total = sum(counts.values())
    parts = " / ".join(f"{BUCKET_ABBR[k]} {counts.get(k, 0)}" for k in BUCKET_ORDER)
    breadth = _breadth_label(crossings_today or {}, crossings_median or {})
    return f"Scan {date:%Y-%m-%d} — {total} flags ({parts}) — breadth: {breadth}"


def send_report(date, report_text: str, buckets: dict,
                 crossings_today: dict | None = None,
                 crossings_median: dict | None = None) -> None:
    """Send the nightly report. Raises on any failure — callers (cmd_scan)
    decide whether to swallow it; the report/log are already persisted by the
    time this is called, so a mail failure never costs the day's data."""
    cfg = _config()
    _require_config(cfg)
    subject = build_subject(date, buckets, crossings_today, crossings_median)
    _send(subject, report_text, _html_wrap(report_text), cfg)


def send_test() -> None:
    """A single test message, so SMTP config can be verified without waiting
    for the scheduled run."""
    cfg = _config()
    _require_config(cfg)
    text = ("This is a test message from the stock scanner's mailer.py.\n"
            "If you received this, SMTP_HOST/PORT/USER/PASSWORD and "
            "REPORT_RECIPIENT are configured correctly.\n")
    _send("Stock scanner — test email", text, _html_wrap(text), cfg)

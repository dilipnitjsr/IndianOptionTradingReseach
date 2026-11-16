#!/usr/bin/env python3
"""
Pull NIFTY backtest data from ICICI Breeze.

Downloads three datasets as CSV, in this order:
  1. NIFTY 50 spot           -> data/nifty_spot_5minute.csv
  2. India VIX               -> data/india_vix_5minute.csv
  3. NIFTY weekly options    -> data/options/NIFTY_<expiry>_<strike>_<CE|PE>.csv
     (calls and puts, strikes around the spot range of each expiry week,
      with volume and open interest)

Setup
  pip install breeze-connect pandas
  export BREEZE_API_KEY=...        (Windows: set BREEZE_API_KEY=...)
  export BREEZE_API_SECRET=...
  export BREEZE_SESSION_TOKEN=...  (fresh token from the Breeze login, valid for the day)

Run
  python pull_nifty_breeze.py                 # everything
  python pull_nifty_breeze.py --only spot     # one dataset: spot | vix | options
  python pull_nifty_breeze.py --start 2024-11-01 --end 2026-11-13

The script is resumable. Files already on disk are skipped, so if it stops at the
daily call cap, run it again the next day with a new session token.

Things to check before the first full run (marked CHECK below):
  - VIX_STOCK_CODE: the ICICI code for India VIX.
  - EXPIRY_WEEKDAY_CHANGES: the weekday of the NIFTY weekly expiry over your period.
  - Run with --only spot first and look at the CSV.
"""
import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from breeze_connect import BreezeConnect

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INTERVAL = "5minute"            # "1minute" | "5minute" | "30minute" | "1day"
START_DATE = "2024-11-01"       # default period: about two years
END_DATE = "2026-11-13"

SPOT_STOCK_CODE = "NIFTY"
SPOT_EXCHANGE = "NSE"
VIX_STOCK_CODE = "INDVIX"       # CHECK: confirm the ICICI stock code for India VIX
OPT_STOCK_CODE = "NIFTY"
OPT_EXCHANGE = "NFO"

STRIKE_STEP = 50                # NIFTY strike spacing
STRIKE_PAD = 300                # points beyond the week's low and high to cover

# Weekly expiry weekday (Mon=0 ... Sun=6), as (effective_from, weekday).
# CHECK: NIFTY weekly expiry was Thursday and moved to Tuesday in 2025.
# Confirm the switch date from the NSE circular for your period.
EXPIRY_WEEKDAY_CHANGES = [
    ("2000-01-01", 3),          # Thursday
    ("2025-09-01", 1),          # Tuesday
]

CALLS_PER_MINUTE = 60           # stay under the Breeze per-minute limit
DAILY_CALL_CAP = 4800           # stay under the Breeze daily limit
MAX_RETRIES = 4
CHUNK_DAYS = 10                 # calendar days per request; keeps 5-minute bars under 1,000

OUT_DIR = Path("data")
OPT_DIR = OUT_DIR / "options"
FAIL_LOG = OUT_DIR / "failed_requests.csv"

# ----------------------------------------------------------------------------
# API helpers
# ----------------------------------------------------------------------------
_calls_made = 0
_last_call = 0.0


class DailyCapReached(Exception):
    pass


def connect():
    try:
        key = os.environ["BREEZE_API_KEY"]
        secret = os.environ["BREEZE_API_SECRET"]
        token = os.environ["BREEZE_SESSION_TOKEN"]
    except KeyError as missing:
        sys.exit(f"Missing environment variable: {missing}")
    breeze = BreezeConnect(api_key=key)
    breeze.generate_session(api_secret=secret, session_token=token)
    return breeze


def iso(d, hhmm):
    """Breeze date format, e.g. 2022-08-15T07:00:00.000Z."""
    return f"{d:%Y-%m-%d}T{hhmm}:00.000Z"


def fetch(breeze, **kwargs):
    """One throttled historical-data call with retries. Returns a list of rows."""
    global _calls_made, _last_call
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if _calls_made >= DAILY_CALL_CAP:
            raise DailyCapReached()
        wait = 60.0 / CALLS_PER_MINUTE - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        _calls_made += 1
        try:
            resp = breeze.get_historical_data_v2(**kwargs)
        except Exception as exc:                      # network or SDK error
            last_error = repr(exc)
        else:
            if isinstance(resp, dict):
                rows = resp.get("Success")
                if rows is not None:
                    return rows
                last_error = str(resp.get("Error"))
                if resp.get("Status") == 200:         # valid call, no data
                    return []
            else:
                last_error = f"unexpected response: {resp!r}"
        time.sleep(2 * attempt)
    raise RuntimeError(last_error or "request failed")


def log_failure(what, error):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not FAIL_LOG.exists()
    with FAIL_LOG.open("a") as fh:
        if new:
            fh.write("time,request,error\n")
        fh.write(f'{datetime.now():%Y-%m-%d %H:%M:%S},"{what}","{error}"\n')


def tidy(rows):
    """Rows -> DataFrame sorted by time, numeric columns converted."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ("open", "high", "low", "close", "volume", "open_interest", "strike_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 1 and 2: index series (spot, VIX)
# ----------------------------------------------------------------------------
def pull_index(breeze, stock_code, out_file, start, end):
    """Pull an index series in date chunks. Resumes from the last saved bar."""
    frames = []
    if out_file.exists():
        old = pd.read_csv(out_file, parse_dates=["datetime"])
        if not old.empty:
            frames.append(old)
            start = max(start, old["datetime"].max().date() + timedelta(days=1))
            print(f"  resuming {out_file.name} from {start}")
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        try:
            rows = fetch(
                breeze,
                interval=INTERVAL,
                from_date=iso(cur, "07:00"),
                to_date=iso(chunk_end, "18:00"),
                stock_code=stock_code,
                exchange_code=SPOT_EXCHANGE,
                product_type="cash",
            )
        except DailyCapReached:
            save_index(frames, out_file)
            raise
        except RuntimeError as exc:
            log_failure(f"{stock_code} {cur}..{chunk_end}", exc)
            rows = []
        df = tidy(rows)
        print(f"  {stock_code} {cur} to {chunk_end}: {len(df)} bars")
        if not df.empty:
            frames.append(df)
        cur = chunk_end + timedelta(days=1)
    return save_index(frames, out_file)


def save_index(frames, out_file):
    if not frames:
        print(f"  no data for {out_file.name}")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    keep = [c for c in ("datetime", "open", "high", "low", "close", "volume") if c in df.columns]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(out_file, index=False)
    print(f"  saved {out_file} ({len(df)} bars)")
    return df[keep]


# ----------------------------------------------------------------------------
# 3: weekly options
# ----------------------------------------------------------------------------
def expiry_weekday(d):
    wd = EXPIRY_WEEKDAY_CHANGES[0][1]
    for eff, weekday in EXPIRY_WEEKDAY_CHANGES:
        if d >= date.fromisoformat(eff):
            wd = weekday
    return wd


def weekly_expiries(trading_days):
    """Expiry dates within the data: the expiry weekday of each week, moved back
    to the previous trading day when that weekday is a holiday."""
    days = sorted(trading_days)
    day_set = set(days)
    expiries = []
    d = days[0]
    while d <= days[-1]:
        if d.weekday() == expiry_weekday(d):
            e = d
            while e not in day_set and e >= days[0]:
                e -= timedelta(days=1)
            if e in day_set and e not in expiries:
                expiries.append(e)
        d += timedelta(days=1)
    return sorted(expiries)


def pull_options(breeze, spot):
    spot = spot.copy()
    spot["day"] = spot["datetime"].dt.date
    trading_days = sorted(spot["day"].unique())
    expiries = weekly_expiries(trading_days)
    print(f"  {len(expiries)} weekly expiries from {expiries[0]} to {expiries[-1]}")
    OPT_DIR.mkdir(parents=True, exist_ok=True)

    prev_expiry = None
    for expiry in expiries:
        # The window in which this expiry is the nearest one.
        if prev_expiry is None:
            win_start = max(trading_days[0], expiry - timedelta(days=7))
        else:
            win_start = prev_expiry + timedelta(days=1)
        week = spot[(spot["day"] >= win_start) & (spot["day"] <= expiry)]
        prev_expiry = expiry
        if week.empty:
            continue
        lo = int((week["low"].min() - STRIKE_PAD) // STRIKE_STEP) * STRIKE_STEP
        hi = int(-((-(week["high"].max() + STRIKE_PAD)) // STRIKE_STEP)) * STRIKE_STEP
        strikes = range(lo, hi + STRIKE_STEP, STRIKE_STEP)
        print(f"  expiry {expiry}: window {win_start} to {expiry}, strikes {lo} to {hi}")

        for strike in strikes:
            for right, tag in (("call", "CE"), ("put", "PE")):
                out_file = OPT_DIR / f"NIFTY_{expiry:%Y%m%d}_{strike}_{tag}.csv"
                if out_file.exists():
                    continue
                try:
                    rows = fetch(
                        breeze,
                        interval=INTERVAL,
                        from_date=iso(win_start, "07:00"),
                        to_date=iso(expiry, "18:00"),
                        stock_code=OPT_STOCK_CODE,
                        exchange_code=OPT_EXCHANGE,
                        product_type="options",
                        expiry_date=iso(expiry, "07:00"),
                        right=right,
                        strike_price=str(strike),
                    )
                except RuntimeError as exc:
                    log_failure(out_file.name, exc)
                    continue
                df = tidy(rows)
                cols = ["datetime", "open", "high", "low", "close", "volume", "open_interest"]
                df = df.reindex(columns=cols) if not df.empty else pd.DataFrame(columns=cols)
                df.insert(1, "expiry", f"{expiry:%Y-%m-%d}")
                df.insert(2, "strike", strike)
                df.insert(3, "right", tag)
                df.to_csv(out_file, index=False)   # empty file marks "no data", so it is not retried
        print(f"    done, {_calls_made} calls used this run")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pull NIFTY spot, India VIX and weekly options from ICICI Breeze.")
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--end", default=END_DATE)
    ap.add_argument("--only", choices=["spot", "vix", "options"])
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    breeze = connect()
    spot_file = OUT_DIR / f"nifty_spot_{INTERVAL}.csv"
    vix_file = OUT_DIR / f"india_vix_{INTERVAL}.csv"
    try:
        if args.only in (None, "spot"):
            print("NIFTY spot")
            pull_index(breeze, SPOT_STOCK_CODE, spot_file, start, end)
        if args.only in (None, "vix"):
            print("India VIX")
            pull_index(breeze, VIX_STOCK_CODE, vix_file, start, end)
        if args.only in (None, "options"):
            print("NIFTY weekly options")
            if not spot_file.exists():
                sys.exit("Spot file not found. Run with --only spot first.")
            spot = pd.read_csv(spot_file, parse_dates=["datetime"])
            spot = spot[(spot["datetime"].dt.date >= start) & (spot["datetime"].dt.date <= end)]
            pull_options(breeze, spot)
    except DailyCapReached:
        print(f"\nStopped at the daily cap of {DAILY_CALL_CAP} calls. "
              "Run again tomorrow with a new session token; finished files are skipped.")
        return
    print(f"\nFinished. {_calls_made} calls used. Failures, if any, are in {FAIL_LOG}.")


if __name__ == "__main__":
    main()

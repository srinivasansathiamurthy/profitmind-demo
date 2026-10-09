"""Upload Dominick's week decode table (manual Section 8, "Week's Decode Table") to GCS as JSON.
Note: sas file has issues. More than just reading off of the documentation with modern day vlms.

Usage:  BUCKET=my-bucket python upload_dominick_fine_foods_date_data.py
Writes: gs://$BUCKET/raw/week_decode/week_decode.json   (--no-clobber: never overwrites)

Weeks 1-400, 7 days each (Thu-Wed); week 1 = 1989-09-14..1989-09-20 (anchors checked
against the manual: weeks 1, 16, 259, 338, 400). `special_event` is as printed in the
manual; a few labels (Easter, Presidents Day, one Halloween) do not sit in the calendar
week of the real holiday, so don't treat it as a calendar-accurate holiday flag.
"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta

WEEK1_START = date(1989, 9, 14)
N_WEEKS = 400

SPECIAL_EVENTS = {
    7: "Halloween", 11: "Thanksgiving", 15: "Christmas", 16: "New-Year", 23: "Presidents Day",
    28: "Easter", 37: "Memorial Day", 42: "4th of July", 51: "Labor Day",
    59: "Halloween", 63: "Thanksgiving", 67: "Christmas", 68: "New-Year", 75: "Presidents Day",
    81: "Easter", 89: "Memorial Day", 95: "4th of July", 103: "Labor Day",
    112: "Halloween", 116: "Thanksgiving", 119: "Christmas", 120: "New-Year", 128: "Presidents Day",
    133: "Easter", 141: "Memorial Day", 147: "4th of July", 156: "Labor Day",
    164: "Halloween", 168: "Thanksgiving", 172: "Christmas", 173: "New-Year", 180: "Presidents Day",
    185: "Easter", 194: "Memorial Day", 199: "4th of July", 208: "Labor Day",
    216: "Halloween", 220: "Thanksgiving", 224: "Christmas", 225: "New-Year", 232: "Presidents Day",
    238: "Easter", 246: "Memorial Day", 251: "4th of July", 260: "Labor Day",
    268: "Halloween", 272: "Thanksgiving", 276: "Christmas", 277: "New-Year", 284: "Presidents Day",
    289: "Easter", 298: "Memorial Day", 303: "4th of July", 312: "Labor Day",
    320: "Halloween", 324: "Thanksgiving", 328: "Christmas", 329: "New-Year", 336: "Presidents Day",
    341: "Easter", 350: "Memorial Day", 356: "4th of July", 364: "Labor Day",
    372: "Halloween", 377: "Thanksgiving", 380: "Christmas", 381: "New-Year", 389: "Presidents Day",
    393: "Easter",
}


def build_week_decode():
    rows = []
    for week in range(1, N_WEEKS + 1):
        start = WEEK1_START + timedelta(days=7 * (week - 1))
        rows.append({
            "week": week,
            "start": start.isoformat(),
            "end": (start + timedelta(days=6)).isoformat(),
            "special_event": SPECIAL_EVENTS.get(week),
        })
    return rows


def main():
    bucket = os.environ.get("BUCKET")
    if not bucket:
        sys.exit("Set BUCKET, e.g. BUCKET=my-bucket python upload_dominick_fine_foods_date_data.py")

    rows = build_week_decode()
    assert rows[0]["start"] == "1989-09-14" and rows[-1]["end"] == "1997-05-14"

    dest = f"gs://{bucket}/raw/week_decode/week_decode.json"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "week_decode.json")
        with open(path, "w") as f:
            json.dump(rows, f, indent=1)
        subprocess.run(["gcloud", "storage", "cp", "--no-clobber", path, dest], check=True)
    print(f"Uploaded {len(rows)} weeks -> {dest}")


if __name__ == "__main__":
    main()
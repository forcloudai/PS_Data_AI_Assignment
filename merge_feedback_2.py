import re
from pathlib import Path

import pandas as pd

UPLOAD_DIR = Path("feedback_2")
OUTPUT_FILE = Path("combined_datasets/merged_feedback_2.csv")

# Sheets to leave out (none by default; see note about Penrith "Jun 26" below)
SKIP_SHEETS = set()  # e.g. {("Penrith_Feedback.xlsx", "Jun 26")}

# ---------------------------------------------------------------- column names
# Every header variant seen in the files -> the 4 standard columns
COLUMN_ALIASES = {
    "date": "Date",
    "department": "Department",
    "dept": "Department",
    "rating": "Rating/Score",
    "score": "Rating/Score",
    "comment": "Comment",
    "comments": "Comment",
    "notes": "Comment",
    "feedback": "Comment",
}
FINAL_COLS = ["Date", "Department", "Rating/Score", "Comment"]

# ---------------------------------------------------------------------- suburbs
# File-name prefix -> suburb. "IPS" is an abbreviation, so it is expanded here.
FILENAME_SUBURB = {"ips": "Ipswich"}
# North QLD file holds two suburbs, so it is split using its "Site" column
SITE_SUBURB = {
    "tsv": "Townsville", "townsville": "Townsville",
    "rok": "Rockhampton", "rocky": "Rockhampton", "rockhampton": "Rockhampton",
}

# ------------------------------------------------------------------------ dates
# Text dates are day-first (dd/mm/yyyy) except the Liverpool Q3 file, which is
# month-first (mm/dd/yyyy): it holds Jul-Sep 2026 but its first part only ranges 7-9.
MONTH_FIRST_FILES = {"liverpool_q3_2026.csv"}


def parse_dates(series: pd.Series, filename: str) -> pd.Series:
    """Return a datetime Series from real dates, Excel serial numbers or text."""
    def one(v):
        if pd.isna(v):
            return pd.NaT
        if isinstance(v, (int, float)):                       # Excel serial number
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(v), unit="D")
        if isinstance(v, (pd.Timestamp,)) or hasattr(v, "year"):  # real date
            return pd.Timestamp(v)
        s = str(v).strip()
        if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", s):
            return pd.to_datetime(s, dayfirst=filename not in MONTH_FIRST_FILES)
        return pd.NaT                                         # e.g. "early Oct", "Aug?"
    return series.map(one)


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the columns to the 4 standard names and drop everything else."""
    renamed = {c: COLUMN_ALIASES[str(c).strip().lower()]
               for c in df.columns if str(c).strip().lower() in COLUMN_ALIASES}
    out = df.rename(columns=renamed)
    missing = [c for c in FINAL_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {list(df.columns)}")
    keep = out[FINAL_COLS].copy()
    if "Site" in df.columns:
        keep["_site"] = df["Site"]
    return keep


def load_file(path: Path) -> list[pd.DataFrame]:
    """Read one file (all sheets for Excel) into standardised frames."""
    if path.suffix.lower() == ".csv":
        sheets = {"csv": pd.read_csv(path, dtype=object)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype=object)

    prefix = re.split(r"[_\s-]", path.stem)[0].lower()       # "Parramatta", "IPS", ...
    frames = []
    for sheet_name, raw in sheets.items():
        if (path.name, sheet_name) in SKIP_SHEETS:
            continue
        df = standardise(raw)
        df["Date"] = parse_dates(df["Date"], path.name)
        if path.name.lower().startswith("north_qld"):
            df["Suburb"] = df.pop("_site").astype(str).str.strip().str.lower().map(SITE_SUBURB)
        else:
            df["Suburb"] = FILENAME_SUBURB.get(prefix, prefix.title())
            df = df.drop(columns=["_site"], errors="ignore")
        frames.append(df)
    return frames


def main():
    files = sorted(p for p in UPLOAD_DIR.iterdir()
                   if p.suffix.lower() in {".csv", ".xlsx"} and not p.name.startswith("~$"))
    frames = [f for p in files for f in load_file(p)]
    merged = pd.concat(frames, ignore_index=True)

    # tidy up
    merged["Department"] = merged["Department"].astype("string").str.strip()
    merged["Rating/Score"] = pd.to_numeric(merged["Rating/Score"]).astype("Int64")
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")   # unparseable dates stay blank
    merged = merged[FINAL_COLS + ["Suburb"]]

    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(merged)} rows to {OUTPUT_FILE}")
    print(merged["Suburb"].value_counts(dropna=False))
    print("Blank dates:", merged["Date"].isna().sum())


if __name__ == "__main__":
    main()

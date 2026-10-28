import glob
import os
import re
from datetime import date, datetime

import pandas as pd

INPUT_DIR = "feedback_3"
OUTPUT_PATH = "combined_datasets/merged_feedback_3.csv"

# Column-name variants found across the files -> standard names
ALIASES = {
    "Date":              {"date", "date received"},
    "Department":        {"department", "dept"},
    "Rating/Score":      {"rating", "score (1-5)", "score (out of 10)", "how was it"},
    "Comment":           {"comment", "comments", "customer comments", "customer feedback"},
}
FINAL_COLS = ["Suburb", "Date", "Department", "Rating/Score", "Comment"]

# Sanity window for the feedback period; anything outside is junk/test data
VALID_START, VALID_END = date(2025, 10, 1), date(2026, 9, 30)
# Sites that closed: dates after closure are year typos (e.g. 2026 instead of 2025)
SITE_CLOSED = {"Shepparton": date(2026, 3, 31)}

DATE_FORMATS = ["%d/%m/%Y", "%d-%b-%y", "%Y-%m-%d"]


def norm(x):
    return str(x).strip().lower()


def suburb_from_filename(path):
    """'Shepparton_feedback_FINAL_-_site_closed.xlsx' -> 'Shepparton'"""
    return re.match(r"[A-Za-z]+", os.path.basename(path)).group(0).title()


def parse_date(x):
    if isinstance(x, (datetime, pd.Timestamp)):
        return None if pd.isna(x) else x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(x.strip(), fmt).date()
            except ValueError:
                pass
    return None


def clean_rating(x):
    if pd.isna(x):
        return None
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x).strip()


def find_blocks(raw):
    """Locate header rows (a sheet can contain several tables, e.g. a template
    change mid-sheet) and return one standardised DataFrame per table."""
    header_rows = []
    for i, row in raw.iterrows():
        cells = {norm(v) for v in row if pd.notna(v)}
        if all(cells & names for names in ALIASES.values()):
            header_rows.append(i)

    blocks = []
    for n, h in enumerate(header_rows):
        end = header_rows[n + 1] if n + 1 < len(header_rows) else len(raw)
        body = raw.iloc[h + 1:end].copy()
        mapping = {}
        for col_idx, v in raw.iloc[h].items():
            for std, names in ALIASES.items():
                if pd.notna(v) and norm(v) in names:
                    mapping[std] = col_idx
        # de-duplicate exact repeat submissions (only if rows carry identifying columns)
        dropped_dupes = 0
        if body.shape[1] > 4:
            before = len(body)
            body = body.dropna(how="all").drop_duplicates()
            dropped_dupes = before - len(body)
        out = pd.DataFrame({std: body[idx] for std, idx in mapping.items()})
        blocks.append((out, dropped_dupes))
    return blocks


def load_file(path):
    if path.lower().endswith(".csv"):
        try:
            raw = pd.read_csv(path, header=None, dtype=object, encoding="utf-8")
        except UnicodeDecodeError:
            raw = pd.read_csv(path, header=None, dtype=object, encoding="cp1252")
        sheets = {"csv": raw}
    else:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    return sheets


frames, log = [], []
for path in sorted(glob.glob(os.path.join(INPUT_DIR, "*"))):
    if not path.lower().endswith((".csv", ".xlsx")):
        continue
    suburb = suburb_from_filename(path)
    for sheet, raw in load_file(path).items():
        blocks = find_blocks(raw)
        if not blocks:
            log.append(f"SKIPPED  {os.path.basename(path)} [{sheet}] - not row-level feedback")
            continue
        for df, dupes in blocks:
            n_in = len(df)
            df["Date"] = df["Date"].map(parse_date)
            # fix year typos on closed sites
            if suburb in SITE_CLOSED:
                closed = SITE_CLOSED[suburb]
                df["Date"] = df["Date"].map(
                    lambda d: d.replace(year=d.year - 1) if d and d > closed else d)
            ok = df["Date"].map(lambda d: d is not None and VALID_START <= d <= VALID_END)
            df = df[ok].copy()
            df["Rating/Score"] = df["Rating/Score"].map(clean_rating)
            df["Suburb"] = suburb
            frames.append(df[FINAL_COLS])
            log.append(f"{os.path.basename(path)} [{sheet}]: {n_in} rows read, "
                       f"{n_in - len(df)} invalid/test/footer dropped, {dupes} duplicates dropped, "
                       f"{len(df)} kept")

merged = pd.concat(frames, ignore_index=True)
merged = merged.sort_values(["Suburb", "Date"], kind="stable").reset_index(drop=True)
merged["Date"] = pd.to_datetime(merged["Date"]).dt.strftime("%Y-%m-%d")
merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n".join(log))
print(f"\nMerged rows: {len(merged)}")
print(merged.groupby("Suburb").size())

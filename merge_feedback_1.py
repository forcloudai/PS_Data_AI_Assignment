"""
Merge all dealership feedback spreadsheets into one CSV (feedback1.csv).

Output columns: Date, Department, Rating, Comment, Suburb
  - Date       -> ISO format (YYYY-MM-DD)
  - Department -> standardised to Sales / Service / Parts
  - Rating     -> integer 1-5, where 5 = Excellent and 1 = Poor
  - Comment    -> free text (blank if none)
  - Suburb     -> taken from each file's name
"""
import re
import glob
import os

import pandas as pd

INPUT_DIR = "feedback_1"          # Change to the feedback_1 folder
OUTPUT_FILE = "combined_datasets/merged_feedback_1.csv"

# Files to leave out. Chermside_Q1_2026_FINAL.xlsx is fully contained in
# Chermside_Q1_2026_FINAL_v2.xlsx (same 112 rows + 6 extra), so loading both
# would duplicate 112 responses.
SKIP_FILES = {"Chermside_Q1_2026_FINAL.xlsx"}

# Filename-derived suburb names that need tidying
SUBURB_FIXES = {"Boxhill": "Box Hill"}

# ---------------------------------------------------------------- helpers --

def suburb_from_filename(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    first = re.split(r"[_\s\-]+", stem)[0].title()
    return SUBURB_FIXES.get(first, first)


def clean_department(value):
    """Map the many department spellings onto Sales / Service / Parts."""
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if v in {"s", "sales", "new cars", "used cars"}:
        return "Sales"
    if v in {"sv", "service", "service dept", "svc", "workshop"}:
        return "Service"
    if v in {"p", "parts", "parts & accessories", "parts counter", "p&a"}:
        return "Parts"
    return str(value).strip()          # unknown label: keep, so it is visible


SATISFACTION = {
    "very dissatisfied": 1, "dissatisfied": 2, "neutral": 3,
    "satisfied": 4, "very satisfied": 5,
}


def stars_to_number(value):
    return str(value).count("\u2605") if pd.notna(value) else None   # count of filled stars


def standardise(df, date, dept, rating, comment, day_first=True):
    """Return a frame with the 4 standard columns from the given source columns."""
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date], dayfirst=day_first, errors="coerce").dt.strftime("%Y-%m-%d"),
        "Department": df[dept].map(clean_department),
        "Rating": df[rating],
        "Comment": df[comment].astype("string").str.strip(),
    })
    return out


def find_header_row(raw, first_cell="Date"):
    """Index of the row whose first cell is `first_cell` (headers are not always on row 1)."""
    mask = raw.iloc[:, 0].astype(str).str.strip().eq(first_cell)
    return mask.idxmax() if mask.any() else None


# ---------------------------------------------------------------- loaders --

def load_ballarat(path):
    df = pd.read_excel(path, header=1)                      # 2-row header: group row + column row
    df["Rating"] = df["How satisfied?"].str.strip().str.lower().map(SATISFACTION)
    return standardise(df, "Date", "Dept", "Rating", "Comments")


def load_bendigo(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Rating"] = df["Stars"].map(stars_to_number)
    return standardise(df, "Date", "Dept", "Rating", "What did they say?")   # dates like '2 Oct 25'


def load_boxhill(path):
    df = pd.read_csv(path)
    return standardise(df, "submitted_at", "department", "csat", "verbatim", day_first=False)


def load_chatswood(path):
    df = pd.read_csv(path, sep=";", dtype=str, keep_default_na=False)      # semicolon-delimited
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Comment"] = df["Comment"].replace("", pd.NA)
    return standardise(df, "Date", "Dept", "Score", "Comment")


def load_chermside(path):
    df = pd.read_excel(path)
    return standardise(df, "Date", "Department", "Rating", "Comments")


def load_dandenong(path):
    xl = pd.ExcelFile(path)
    parts = []
    for sheet in xl.sheet_names:                            # one sheet per month
        raw = xl.parse(sheet, header=None)
        if raw.empty:                                       # 'Sheet3' is empty
            continue
        header_row = find_header_row(raw)                   # header row moves (May has an extra note row)
        df = xl.parse(sheet, header=header_row)
        parts.append(standardise(df, "Date", "Dept", "Rating (1-5)", "Comments"))
    return pd.concat(parts, ignore_index=True)


def load_essendon(path):
    xl = pd.ExcelFile(path)
    parts = []
    for sheet in xl.sheet_names:                            # one sheet per department
        df = xl.parse(sheet)
        df["Department"] = sheet                            # the department only exists as the sheet name
        parts.append(standardise(df, "Date", "Department", "Score", "Customer said"))
    return pd.concat(parts, ignore_index=True)


def load_frankston(path):
    df = pd.read_excel(path)
    # Drop blank rows, monthly 'Total ...' rows and the repeated header row:
    # real responses are the only rows with a true date value.
    df = df[df["Date"].map(lambda x: isinstance(x, (pd.Timestamp,)) or hasattr(x, "year"))].copy()
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    return standardise(df, "Date", "Area", "Rating", "Comments")


def load_geelong(path):
    xl = pd.ExcelFile(path)
    parts = []
    for sheet in xl.sheet_names:                            # '2025' and '2026'
        raw = xl.parse(sheet, header=None)
        df = xl.parse(sheet, header=find_header_row(raw))
        out = standardise(df, "Date", "Department", "Rating", "Feedback")   # d/m/yy strings
        # The 2025 sheet states "1 = Excellent, 5 = Poor"; 2026 uses "1 = Poor, 5 = Excellent".
        # Flip 2025 so the whole file uses 5 = Excellent.
        if "1 = Excellent" in " ".join(raw.iloc[:3, 0].dropna().astype(str)):
            out["Rating"] = 6 - pd.to_numeric(out["Rating"])
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def load_hunter(path):
    raw = pd.read_excel(path, header=None)
    df = pd.read_excel(path, header=find_header_row(raw, first_cell="Dealer"))
    return standardise(df, "Date", "Department", "Rating", "Comments")


# Pick the loader by filename (case-insensitive prefix)
LOADERS = {
    "ballarat": load_ballarat,
    "bendigo": load_bendigo,
    "boxhill": load_boxhill,
    "chatswood": load_chatswood,
    "chermside": load_chermside,
    "dandenong": load_dandenong,
    "essendon": load_essendon,
    "frankston": load_frankston,
    "geelong": load_geelong,
    "hunter": load_hunter,
}

# ------------------------------------------------------------------- main --

def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx")) +
                   glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    frames = []
    for path in files:
        name = os.path.basename(path)
        if name in SKIP_FILES:
            print(f"skipped   {name}")
            continue
        key = re.split(r"[_\s\-]+", name.lower())[0]
        loader = LOADERS.get(key)
        if loader is None:
            print(f"NO LOADER {name}")
            continue
        df = loader(path)
        df["Suburb"] = suburb_from_filename(path)
        print(f"loaded    {name:50s} {len(df):5d} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["Rating"] = pd.to_numeric(merged["Rating"], errors="coerce").astype("Int64")
    merged = merged[["Date", "Department", "Rating", "Comment", "Suburb"]]
    merged = merged.sort_values(["Suburb", "Date"], kind="stable").reset_index(drop=True)

    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(merged)} rows to {OUTPUT_FILE}")
    return merged


if __name__ == "__main__":
    main()

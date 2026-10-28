import pandas as pd

SRC = "combined_datasets/merged_all_feedback.csv"
DST = "combined_datasets/merged_all_feedback_cleaned.csv"

# Read everything as text so nothing gets auto-converted (e.g. blanks to NaN)
df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
rows_in = len(df)

# 1. Remove exact duplicate rows (all 5 columns identical), keeping the first
df = df.drop_duplicates(keep="first").reset_index(drop=True)
dupes_removed = rows_in - len(df)

# 2. Department: standardise to Service / Sales / Parts
dept_map = {
    "service": "Service", "service dept": "Service", "svc": "Service",
    "workshop": "Service",
    "sales": "Sales", "sales - new": "Sales", "sales - used": "Sales",
    "new cars": "Sales", "used cars": "Sales",
    "parts": "Parts", "parts & accessories": "Parts",
    "parts counter": "Parts", "p&a": "Parts",
}
key = df["Department"].str.strip().str.lower()
unmapped = sorted(set(key) - set(dept_map))
assert not unmapped, f"Unmapped departments: {unmapped}"
df["Department"] = key.map(dept_map)

# 3. Rating/Score: put everything on the 1-5 scale
#    - Tamworth text labels -> numbers (ASSUMPTION: edit as needed)
#    - Ringwood 6-10 scores -> 1-5 by halving, rounding .5 up
#    - blanks stay blank (no value is invented)
text_map = {"Good": 5, "OK": 3, "Bad": 1}

def fix_rating(v):
    v = v.strip()
    if v == "":
        return ""
    if v in text_map:
        return str(text_map[v])
    n = int(v)
    if n > 5:
        return str(int(n / 2 + 0.5))   # 6->3, 7->4, 8->4, 9->5, 10->5
    return str(n)

df["Rating/Score"] = df["Rating/Score"].map(fix_rating)

# 4. Trim stray whitespace in every column (comment wording is left untouched)
for col in df.columns:
    df[col] = df[col].str.strip()

# 5. Write out
df.to_csv(DST, index=False)

print(f"Rows in: {rows_in}  |  duplicates removed: {dupes_removed}  |  rows out: {len(df)}")
print(df["Department"].value_counts().to_string())
print(df["Rating/Score"].value_counts(dropna=False).sort_index().to_string())

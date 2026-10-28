import pandas as pd

# Input files (in the order they should be stacked)
files = [
    "combined_datasets/merged_feedback_1.csv",
    "combined_datasets/merged_feedback_2.csv",
    "combined_datasets/merged_feedback_3.csv",
]

frames = []
for path in files:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    # File 1 calls this column "Rating"; files 2 and 3 call it "Rating/Score".
    # Standardise the name so the values land in one column.
    df = df.rename(columns={"Rating": "Rating/Score"})
    frames.append(df)

# Stack all rows. concat aligns on column NAMES, so file 3's different
# column order is handled automatically.
merged = pd.concat(frames, ignore_index=True)

# Fix a consistent column order
merged = merged[["Date", "Department", "Rating/Score", "Comment", "Suburb"]]

merged.to_csv("combined_datasets/merged_all_feedback.csv", index=False)

print("Rows per file:", [len(f) for f in frames])
print("Merged shape:", merged.shape)

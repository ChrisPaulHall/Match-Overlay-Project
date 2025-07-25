import pandas as pd
from rapidfuzz import process, fuzz

# 1) Load your two files
df_current = pd.read_csv("OSsource_info.csv")    # your main CSV
df_2018    = pd.read_csv("2018_net_worth.csv")     # the 2018 file, must contain a networth column

# 2) Normalize names to a key for matching
def make_key(df, first_col, last_col):
    return (
        df[first_col].str.lower().str.strip()
        + " "
        + df[last_col].str.lower().str.strip()
    )

df_current["name_key"] = make_key(df_current, "first_name", "last_name")
df_2018   ["name_key"] = make_key(df_2018,    "first_name", "last_name")

merged = df_current.merge(
    df_2018[["name_key","net_worth"]],
    on="name_key",
    how="left",
    validate="one_to_one"
).rename(columns={"net_worth":"networth_2018"})

net_map = dict(zip(df_2018["name_key"], df_2018["net_worth"]))
choices = list(net_map.keys())

def find_networth(key, choices, cutoff=85):
    match, score, _ = process.extractOne(key, choices, scorer=fuzz.token_sort_ratio)
    return net_map[match] if score>=cutoff else None

merged["networth_2018"] = merged.apply(
    lambda r: r["networth_2018"] if pd.notna(r["networth_2018"])
              else find_networth(r["name_key"], choices),
    axis=1
)

# 5) Save out the updated CSV
merged.to_csv("current_with_networth.csv", index=False)
print("Wrote current_with_networth.csv — networth_2018 filled where names matched.")

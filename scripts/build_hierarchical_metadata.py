import pandas as pd

# Load files
meta = pd.read_csv("data/ip102/metadata.csv")
tax = pd.read_csv("data/ip102/cleaned_taxonomy.csv")

# Merge taxonomy into metadata
df = meta.merge(tax, on="label", how="left")

# Build numeric ids for hierarchy levels
def make_ids(series):
    vals = series.fillna("UNKNOWN").astype(str)
    unique_vals = sorted(vals.unique())
    mapping = {v: i for i, v in enumerate(unique_vals)}
    return vals.map(mapping), mapping

df["species_id"], species_map = make_ids(df["species"])
df["genus_id"], genus_map = make_ids(df["genus"])
df["family_id"], family_map = make_ids(df["family"])
df["order_id"], order_map = make_ids(df["order"])

# Save merged metadata
df.to_csv("data/ip102/metadata_hierarchical.csv", index=False)

print("Saved: data/ip102/metadata_hierarchical.csv")
print("Num species ids:", len(species_map))
print("Num genus ids:", len(genus_map))
print("Num family ids:", len(family_map))
print("Num order ids:", len(order_map))

print("\nOrder mapping:")
for k, v in order_map.items():
    print(v, k)
import pandas as pd
from pygbif import species

df = pd.read_csv("data/ip102/classes_clean.csv")

taxonomy_rows = []

for _, row in df.iterrows():

    name = row["class_name"]

    try:
        result = species.name_backbone(name=name)

        taxonomy_rows.append({
            "label": row["label"],
            "class_name": name,
            "species": result.get("species"),
            "genus": result.get("genus"),
            "family": result.get("family"),
            "order": result.get("order")
        })

    except Exception:
        taxonomy_rows.append({
            "label": row["label"],
            "class_name": name,
            "species": None,
            "genus": None,
            "family": None,
            "order": None
        })

taxonomy_df = pd.DataFrame(taxonomy_rows)

taxonomy_df.to_csv("data/ip102/taxonomy.csv", index=False)

print("Saved data/ip102/taxonomy.csv")
print(taxonomy_df.head())
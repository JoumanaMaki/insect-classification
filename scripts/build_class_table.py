import pandas as pd

rows = []

with open("data/ip102/classes.txt", "r") as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        class_id = int(parts[0]) - 1
        class_name = parts[1]

        rows.append({
            "label": class_id,
            "class_name": class_name
        })

df = pd.DataFrame(rows)

print(df.head())
print(df.tail())

df.to_csv("data/ip102/classes_clean.csv", index=False)

print("Saved data/ip102/classes_clean.csv")
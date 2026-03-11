import pandas as pd

df = pd.read_csv("data/ip102/classes_clean.csv")

df[["class_name"]].to_csv("data/ip102/pest_names.csv", index=False)

print("Saved pest_names.csv")
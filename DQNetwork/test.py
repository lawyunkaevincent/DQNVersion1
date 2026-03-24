import pandas as pd

df = pd.read_csv("imitation_dataset.csv")

# How often was DEFER the chosen action?
chosen = df[df["chosen"] == 1].copy()
defer_rate = chosen["candidate_is_defer"].mean()

print("chosen decisions:", len(chosen))
print("defer chosen rate:", defer_rate)

# Compare counts
print(chosen["candidate_is_defer"].value_counts(dropna=False))
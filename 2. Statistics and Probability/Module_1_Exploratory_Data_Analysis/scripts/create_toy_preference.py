import pandas as pd
import numpy as np
import os

# Make sure the folder exists
os.makedirs("data", exist_ok=True)

# Sample names
names = ["Aisha", "Ben", "Clara", "David", "Ella", "Farhan", "Grace", "Hiro", "Isla", "Jack"]

# Generate a fair amount of data (100 entries)
n = 100
toy_data = pd.DataFrame({
    "Name": np.random.choice(names, n),
    "Likes Cars (1=Yes, 0=No)": np.random.choice([0, 1], n, p=[0.4, 0.6])  # 60% like cars
})
toy_data.to_csv('./data/toy_preferences.csv', index=False)
print("Dataset saved as data/toy_preferences.csv with", n, "entries")

# Optional: Generate 50/50 preference data
toy_data_50_50 = pd.DataFrame({
    "Name": np.random.choice(names, n),
    "Likes Cars (1=Yes, 0=No)": np.random.choice([0, 1], n, p=[0.5, 0.5])  # 50/50 split
})
toy_data_50_50.to_csv('./data/toy_preferences_50_50.csv', index=False)
print("50/50 dataset saved as data/toy_preferences_50_50.csv with", n, "entries")
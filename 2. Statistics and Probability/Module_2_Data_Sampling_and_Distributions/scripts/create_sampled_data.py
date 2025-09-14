import pandas as pd
import numpy as np
import os

# Make sure the folder exists
os.makedirs("data", exist_ok=True)

# Generate a fair amount of data (50 entries)
n = 50
np.random.seed(42)  # For reproducibility
data = pd.DataFrame({
    "ID": range(1, n + 1),
    "Score": np.random.normal(85, 5, n).round()  # Mean 85, std 5
})
data.to_csv('data/sampled_data.csv', index=False)
print("Dataset saved as data/sampled_data.csv with {} entries".format(n))
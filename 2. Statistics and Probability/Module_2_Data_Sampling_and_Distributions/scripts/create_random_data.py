import pandas as pd
import numpy as np
import os

# Make sure the folder exists
os.makedirs("data", exist_ok=True)

# Generate a fair amount of data (100 entries)
n = 100
np.random.seed(42)  # For reproducibility
data = pd.DataFrame({
    "ID": range(1, n + 1),
    "Age": np.random.randint(9, 14, n),  # Ages 9 to 13
    "Score": np.random.randint(70, 100, n)  # Scores 70 to 99
})
data.to_csv('./data/random_data.csv', index=False)
print("Dataset saved as data/random_data.csv with {} entries".format(n))
import pandas as pd
import numpy as np
import os

# Make sure the folder exists
os.makedirs("data", exist_ok=True)

# Generate a small dataset based on the provided values
small_ages = [22, 38, 26, 35, 54, 2, 27, 14]
small_data = pd.DataFrame({'Age': small_ages})
small_data.to_csv('./data/titanic_ages.csv', index=False)
print("Small dataset saved as data/titanic_ages.csv")

# Generate a larger dataset for more robust analysis
np.random.seed(42)  # For reproducibility
ages = np.concatenate([np.random.normal(30, 10, 800), np.random.normal(60, 5, 200)])  # Mostly 20-40, some older
large_data = pd.DataFrame({'Age': ages})
large_data.to_csv('./data/titanic_ages_large.csv', index=False)
print("Large dataset saved as data/titanic_ages_large.csv")
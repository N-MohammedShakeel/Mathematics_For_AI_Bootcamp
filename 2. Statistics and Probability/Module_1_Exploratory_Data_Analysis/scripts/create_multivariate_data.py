import pandas as pd
import numpy as np
import os

# Make sure the folder exists
os.makedirs("data", exist_ok=True)

# Sample names
names = ["Aisha", "Ben", "Clara", "David", "Ella", "Farhan", "Grace", "Hiro", "Isla", "Jack"]

# Generate a fair amount of data (100 entries)
n = 100
np.random.seed(42)  # For reproducibility
data = pd.DataFrame({
    "Name": np.random.choice(names, n),
    "Age": np.random.randint(8, 12, n),
    "Number of Toys": np.random.randint(1, 10, n),
    "Height (cm)": np.random.normal(130, 5, n).round()  # Mean 130cm, std 5cm
})
data.to_csv('./data/multivariate_data.csv', index=False)
print("Dataset saved as data/multivariate_data.csv with {} entries".format(n))
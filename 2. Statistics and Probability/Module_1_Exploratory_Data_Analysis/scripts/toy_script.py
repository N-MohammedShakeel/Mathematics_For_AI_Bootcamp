import pandas as pd
import numpy as np
import os
import random

# Make sure the folder exists
os.makedirs("data", exist_ok=True)

# Sample lists
names = ["Aisha", "Ben", "Clara", "David", "Ella", "Farhan", "Grace", "Hiro", "Isla", "Jack"]
toys = ["Car", "Doll", "Block", "Puzzle", "Ball", "Train", "Robot", "Teddy", "Lego", "Kite"]

# Generate large dataset
n = 100
toy_data = pd.DataFrame({
    "ID": range(1, n+1),
    "Name": np.random.choice(names, n),
    "Favorite Toy": np.random.choice(toys, n),
    "Number of Toys": np.random.randint(1, 21, n),   # between 1 and 20 toys
    "Price per Toy": np.round(np.random.uniform(5, 50, n), 2), # random price
    "Is Gifted": np.random.choice([True, False], n, p=[0.3, 0.7])  # 30% gifted
})

# Save to CSV
toy_data.to_csv("./data/toy_data.csv", index=False)
print("Large toy dataset saved with", n, "rows -> data/toy_data_large.csv")

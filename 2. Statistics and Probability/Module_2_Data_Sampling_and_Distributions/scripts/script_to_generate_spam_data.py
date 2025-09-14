import pandas as pd

# Define the custom binary dataset
data = {
    'Email ID': [1, 2, 3, 4, 5],
    'Has Link (1=Yes)': [1, 0, 1, 0, 1],
    'Is Spam (1=Yes)': [1, 0, 1, 0, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure the 'data' folder exists, create it if it doesn't
import os
if not os.path.exists('data'):
    os.makedirs('data')

# Save to CSV in the data folder
df.to_csv('data/spam_data.csv', index=False)

print("spam_data.csv has been generated in the 'data' folder.")
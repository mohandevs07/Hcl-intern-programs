import pandas as pd

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)

data = {
    "Age": [20, 30, 40, 50, 60],
    "Salary": [
        20000,
        40000,
        60000,
        80000,
        100000
    ]
}

df = pd.DataFrame(data)

print("===== ORIGINAL DATA =====")
print(df)

# Min-Max Scaling
minmax_scaler = MinMaxScaler()

minmax_result = minmax_scaler.fit_transform(df)

minmax_df = pd.DataFrame(
    minmax_result,
    columns=df.columns
)

# Standardization
standard_scaler = StandardScaler()

standard_result = standard_scaler.fit_transform(df)

standard_df = pd.DataFrame(
    standard_result,
    columns=df.columns
)

print("\n===== MIN-MAX SCALING =====")
print(minmax_df)

print("\n===== STANDARDIZATION =====")
print(standard_df)

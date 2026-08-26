import pandas as pd
import numpy as np

# Sample dataset
data = {
    "Age": [22, 25, np.nan, 30, 27],
    "Salary": [25000, 30000, 35000, np.nan, 40000],
    "City": [
        "Chennai",
        "Coimbatore",
        "Chennai",
        "Madurai",
        np.nan
    ]
}

df = pd.DataFrame(data)

print("===== ORIGINAL DATA =====")
print(df)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Mean
mean_df = df.copy()

mean_df["Age"] = mean_df["Age"].fillna(
    mean_df["Age"].mean()
)

mean_df["Salary"] = mean_df["Salary"].fillna(
    mean_df["Salary"].mean()
)

# Median
median_df = df.copy()

median_df["Age"] = median_df["Age"].fillna(
    median_df["Age"].median()
)

median_df["Salary"] = median_df["Salary"].fillna(
    median_df["Salary"].median()
)

# Mode
mode_df = df.copy()

mode_df["City"] = mode_df["City"].fillna(
    mode_df["City"].mode()[0]
)

print("\n===== MEAN IMPUTATION =====")
print(mean_df)

print("\n===== MEDIAN IMPUTATION =====")
print(median_df)

print("\n===== MODE IMPUTATION =====")
print(mode_df)

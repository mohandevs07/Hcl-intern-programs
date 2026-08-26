import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


# 1. Create raw dataset


data = {
    "Age": [22, 25, np.nan, 30, 27],
    "Salary": [
        25000,
        30000,
        35000,
        40000,
        200000
    ],
    "City": [
        "Chennai",
        "Madurai",
        "Chennai",
        "Coimbatore",
        np.nan
    ]
}

df = pd.DataFrame(data)

print("===== ORIGINAL DATA =====")
print(df)


# 2. Missing values


df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Salary"] = df["Salary"].fillna(
    df["Salary"].median()
)

df["City"] = df["City"].fillna(
    df["City"].mode()[0]
)

print("\n===== AFTER MISSING VALUE TREATMENT =====")
print(df)


# 3. IQR Outlier Removal


Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["Salary"] >= lower) &
    (df["Salary"] <= upper)
]

print("\n===== AFTER OUTLIER REMOVAL =====")
print(df)

# 4. One-Hot Encoding


df = pd.get_dummies(
    df,
    columns=["City"],
    dtype=int
)

print("\n===== AFTER ENCODING =====")
print(df)


# 5. Standardization


scaler = StandardScaler()

numeric_columns = [
    "Age",
    "Salary"
]

df[numeric_columns] = scaler.fit_transform(
    df[numeric_columns]
)

print("\n===== FINAL PREPROCESSED DATA =====")
print(df)

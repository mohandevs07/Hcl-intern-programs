import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Salary": [
        25000, 28000, 30000, 32000, 35000,
        36000, 38000, 40000, 42000, 200000
    ]
}

df = pd.DataFrame(data)

print("===== ORIGINAL DATA =====")
print(df)

# Q1 and Q3
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

# IQR
IQR = Q3 - Q1

# Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Detect outliers
outliers = df[
    (df["Salary"] < lower_bound) |
    (df["Salary"] > upper_bound)
]

print("\n===== OUTLIERS =====")
print(outliers)

# Remove outliers
clean_df = df[
    (df["Salary"] >= lower_bound) &
    (df["Salary"] <= upper_bound)
]

print("\n===== AFTER OUTLIER REMOVAL =====")
print(clean_df)

# Visualization
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.boxplot(df["Salary"])
plt.title("Before Outlier Removal")

plt.subplot(1, 2, 2)
plt.boxplot(clean_df["Salary"])
plt.title("After Outlier Removal")

plt.tight_layout()
plt.show()

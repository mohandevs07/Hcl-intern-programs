import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder
)

# Create dataset
df = pd.DataFrame({
    "City": [
        "Chennai",
        "Coimbatore",
        "Madurai",
        "Chennai"
    ]
})

print("ORIGINAL DATA")
print(df)


# -------------------------
# Label Encoding
# -------------------------

label_encoder = LabelEncoder()

label_encoded = label_encoder.fit_transform(df["City"])

print("\nLABEL ENCODING")
print(label_encoded)


# -------------------------
# One-Hot Encoding
# -------------------------

one_hot_encoder = OneHotEncoder()

one_hot_encoded = one_hot_encoder.fit_transform(
    df[["City"]]
).toarray()

one_hot_df = pd.DataFrame(
    one_hot_encoded,
    columns=one_hot_encoder.get_feature_names_out(["City"])
)

print("\nONE-HOT ENCODING")
print(one_hot_df)

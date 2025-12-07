import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

"""
======================================================
Load Datasets
======================================================
"""
X_train = pd.read_csv("/Users/skygastinel/indeng/secondhand-clothing-sales/data/X_train_clean.csv")
X_test = pd.read_csv("/Users/skygastinel/indeng/secondhand-clothing-sales/data/X_test_clean.csv")
y_train = pd.read_csv("/Users/skygastinel/indeng/secondhand-clothing-sales/data/y_train_clean.csv", header=None).squeeze("columns")

"""
======================================================
Initial Feature Engineering
======================================================
"""
# Brand processing: convert brands into tiers using target encoding
brand_df = X_train.copy()
y_train = pd.to_numeric(y_train, errors="coerce")
brand_df["y"] = y_train

brand_stats = brand_df.groupby("brand_name")["y"].agg(["mean", "count"])
global_mean = y_train.mean()
k = 10

brand_stats["smooth_te"] = (
    brand_stats["count"] * brand_stats["mean"] + k * global_mean
) / (brand_stats["count"] + k)

brand_stats["tier"] = pd.qcut(
    brand_stats["smooth_te"],
    q=5,
    labels=["Budget", "Mid-Market", "Premium", "Luxury", "Ultra-Luxury"]
)

brand_to_tier = brand_stats["tier"].to_dict()

X_train["brand_tier"] = X_train["brand_name"].map(brand_to_tier).fillna("Unknown")
X_test["brand_tier"] = X_test["brand_name"].map(brand_to_tier).fillna("Unknown")

X_train = X_train.drop(columns=["brand_name"])
X_test = X_test.drop(columns=["brand_name"])

# Handle missing shipping times
X_train["ships_missing"] = X_train["usually_ships_within_days"].isna().astype(int)
X_test["ships_missing"] = X_test["usually_ships_within_days"].isna().astype(int)

median_ship = X_train["usually_ships_within_days"].median()
X_train["usually_ships_within_days"] = X_train["usually_ships_within_days"].fillna(median_ship)
X_test["usually_ships_within_days"] = X_test["usually_ships_within_days"].fillna(median_ship)

"""
======================================================
Transformation Functions
======================================================
"""
def transform_product_type(entry):
    if pd.isna(entry):
        return 0
    entry = entry.lower()
    if "jacket" in entry or "coat" in entry or "vest" in entry or "cardigan" in entry or "blazer" in entry or "sweatshirt" in entry or "jumper" in entry or "knitwear" in entry or "puffer" in entry or "caban" in entry or "peacoat" in entry:
        return 1
    if "pant" in entry or "trouser" in entry or "legging" in entry or "jean" in entry or "short" in entry or "tight" in entry or "bermuda" in entry or "skirt" in entry:
        return 2
    if "top" in entry or "shirt" in entry or "blouse" in entry or "corset" in entry:
        return 3
    if "swimsuit" in entry or "bikini" in entry:
        return 4
    if "dress" in entry:
        return 5
    if "jumpsuit" in entry or "romper" in entry:
        return 6
    return 0

def transform_product_season(entry):
    if pd.isna(entry):
        return 0
    entry = entry.lower()
    if "all seasons" in entry:
        return 1
    if "autumn / winter" in entry:
        return 2
    if "spring / summer" in entry:
        return 3
    return 0

def transform_product_material(entry):
    if pd.isna(entry):
        return 0
    entry = entry.lower()
    if "viscose" in entry:
        return 1
    if "cotton" in entry:
        return 2
    if "synthetic" in entry:
        return 3
    if "polyester" in entry:
        return 4
    if "polyamide" in entry:
        return 5
    if "wool" in entry:
        return 6
    if "leather" in entry:
        return 7
    if "silk" in entry:
        return 8
    if "denim" in entry:
        return 9
    if "suede" in entry:
        return 10
    if "khaki" in entry:
        return 11
    if "linen" in entry:
        return 12
    if "cashmere" in entry:
        return 13
    if "velvet" in entry:
        return 14
    if "lace" in entry:
        return 15
    if "fur" in entry:
        return 16
    if "glitter" in entry:
        return 17
    if "tweed" in entry:
        return 18
    if "vinyl" in entry:
        return 19
    if "lycra" in entry:
        return 20
    if "spandex" in entry:
        return 21
    if "shearling" in entry:
        return 22
    if "rubber" in entry:
        return 23
    if "plastic" in entry:
        return 24
    if "cloth" in entry:
        return 25
    if "sponge" in entry:
        return 26
    if 'python' in entry or 'fox' in entry or 'rabbit' in entry or 'water snake' in entry or 'mink' in entry or 'beaver' in entry or 'mongolian lamb' in entry or 'raccoon' in entry or 'crocodile' in entry or 'astrakhan' in entry or 'ostrich' in entry or 'eel' in entry or 'pony' in entry or 'chinchilla' in entry or 'alligator' in entry:
        return 27
    return 0

def transform_color_clean(entry):
    if pd.isna(entry):
        return 0
    entry = entry.lower()
    if "gold" in entry or "silver" in entry or "metallic" in entry:
        return 1
    if "multicolour" in entry:
        return 2
    if "blue" in entry or "navy" in entry:
        return 3
    if "black" in entry:
        return 4
    if "brown" in entry or "beige" in entry or "ecru" in entry or "khaki" in entry or "camel" in entry:
        return 5
    if "green" in entry:
        return 6
    if "pink" in entry:
        return 7
    if "white" in entry:
        return 8
    if "red" in entry or "burgundy" in entry:
        return 9
    if "grey" in entry or "anthracite" in entry:
        return 10
    if "yellow" in entry:
        return 11
    if "orange" in entry:
        return 12
    if "purple" in entry:
        return 13
    if "turquoise" in entry:
        return 14
    return 0

def transform_seller_badge(entry):
    if pd.isna(entry):
        return 0
    entry = entry.lower()
    if "common" in entry:
        return 1
    if "trusted" in entry:
        return 2
    if "expert" in entry:
        return 3
    return 0

def transform_seller_country(entry):
    if pd.isna(entry):
        return 0
    if "United Kingdom" in entry or "Cyprus" in entry or "France" in entry or "Italy" in entry or "Switzerland" in entry or "Belgium" in entry or "Hungary" in entry or "Spain" in entry or "Netherlands" in entry or "Austria" in entry or "Latvia" in entry or "Sweden" in entry or "Finland" in entry or "Germany" in entry or "Poland" in entry or "Romania" in entry or "Denmark" in entry or "Lithuania" in entry or "Portugal" in entry or "Greece" in entry or "Luxembourg" in entry or "Ireland" in entry or "Monaco" in entry or "Bulgaria" in entry or "Czech Republic" in entry or "Slovenia" in entry or "Slovakia" in entry or "Norway" in entry or "Croatia" in entry or "Estonia" in entry or "Albania" in entry or "Malta" in entry or "Liechtenstein" in entry or "Ukraine" in entry or "Belarus" in entry or "Andorra" in entry or "Serbia" in entry or "Isle Of Man" in entry:
        return 1
    if "Indonesia" in entry or "Japan" in entry or "Saudi Arabia" in entry or "Hong Kong" in entry or "Turkey" in entry or "Lebanon" in entry or "Singapore" in entry or "South Korea" in entry or "China" in entry or "Philippines" in entry or "Israel" in entry or "Russia" in entry or "Malaysia" in entry or "Thailand" in entry or "Bahrain" in entry or "Mongolia" in entry or "Taiwan" in entry or "Kazakhstan" in entry or "Qatar" in entry or "Kuwait" in entry or "Macau" in entry or "United Arab Emirates" in entry:
        return 2
    if "United States" in entry or "Canada" in entry or "Mexico" in entry:
        return 3
    if "Brazil" in entry:
        return 4
    if "Nigeria" in entry or "Tunisia" in entry or "The Canary Islands" in entry:
        return 5
    if "Australia" in entry or "New Zealand" in entry:
        return 6
    if "St Barthelemy" in entry or "Guadeloupe" in entry:
        return 7
    return 0

def transform_product_condition(entry):
    if "Never worn, with tag" in entry:
        return 7
    if "Never worn" in entry:
        return 6
    if "Very good condition" in entry:
        return 5
    if "Good condition" in entry:
        return 4
    if "Satisfactory condition" in entry:
        return 3
    if "Fair condition" in entry:
        return 2
    if "Poor condition" in entry:
        return 1
    return 0

"""
======================================================
Apply Transformations
======================================================
"""
# Transform X_train
X_train["cleaned_product_type"] = X_train["product_type"].apply(transform_product_type)
X_train["cleaned_product_season"] = X_train["product_season"].apply(transform_product_season)
X_train["cleaned_product_material"] = X_train["product_material"].apply(transform_product_material)
X_train["cleaned_color_clean"] = X_train["color_clean"].apply(transform_color_clean)
X_train["cleaned_seller_badge"] = X_train["seller_badge"].apply(transform_seller_badge)
X_train["cleaned_seller_country"] = X_train["seller_country"].apply(transform_seller_country)
X_train["cleaned_product_condition"] = X_train["product_condition"].apply(transform_product_condition)

X_train = X_train.drop(columns=["product_type", "product_season", "product_material", "color_clean", "seller_badge", "seller_country", "product_condition"])

# Transform X_test
X_test["cleaned_product_type"] = X_test["product_type"].apply(transform_product_type)
X_test["cleaned_product_season"] = X_test["product_season"].apply(transform_product_season)
X_test["cleaned_product_material"] = X_test["product_material"].apply(transform_product_material)
X_test["cleaned_color_clean"] = X_test["color_clean"].apply(transform_color_clean)
X_test["cleaned_seller_badge"] = X_test["seller_badge"].apply(transform_seller_badge)
X_test["cleaned_seller_country"] = X_test["seller_country"].apply(transform_seller_country)
X_test["cleaned_product_condition"] = X_test["product_condition"].apply(transform_product_condition)

X_test = X_test.drop(columns=["product_type", "product_season", "product_material", "color_clean", "seller_badge", "seller_country", "product_condition"])

"""
======================================================
One-Hot Encoding
======================================================
"""
one_hot_cols = [
    "cleaned_product_type",
    "cleaned_product_season",
    "cleaned_product_material",
    "cleaned_product_condition",
    "cleaned_color_clean",
    "cleaned_seller_country",
    "cleaned_seller_badge",
    "brand_tier",
]

X_train = pd.get_dummies(X_train, columns=one_hot_cols, prefix=one_hot_cols, drop_first=True).astype(int)
X_test = pd.get_dummies(X_test, columns=one_hot_cols, prefix=one_hot_cols, drop_first=True).astype(int)

# Align X_test columns to X_train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

"""
======================================================
Save FULL Data (for CART and Gradient Boosting)
======================================================
"""
print("\n" + "="*70)
print("SAVING DATA FILES")
print("="*70)

print(f"\n1. Saving FULL encoded data (for CART and Gradient Boosting)...")
X_train.to_csv("/Users/skygastinel/indeng/secondhand-clothing-sales/data/X_train_clean_encoded_FULL.csv", index=False)
X_test.to_csv("/Users/skygastinel/indeng/secondhand-clothing-sales/data/X_test_clean_encoded_FULL.csv", index=False)
print(f"   ✓ X_train_FULL: {X_train.shape[0]} rows, {X_train.shape[1]} features")
print(f"   ✓ X_test_FULL: {X_test.shape[0]} rows, {X_test.shape[1]} features")

"""
======================================================
VIF Filtering (for Linear Regression only)
======================================================
"""
print(f"\n2. Applying VIF filtering (for Linear Regression only)...")

vif_col_drop = [
    "cleaned_product_type_1",
    "cleaned_product_season_2",
    "cleaned_product_material_6",
    "cleaned_product_condition_5",
    "cleaned_product_condition_6",
    "cleaned_color_clean_4"
]

X_train_vif = X_train.drop(columns=vif_col_drop)
X_test_vif = X_test.drop(columns=vif_col_drop)

print(f"   ✓ Removed {len(vif_col_drop)} high VIF features (VIF > 10)")
print(f"   ✓ X_train_VIF: {X_train_vif.shape[0]} rows, {X_train_vif.shape[1]} features")
print(f"   ✓ X_test_VIF: {X_test_vif.shape[0]} rows, {X_test_vif.shape[1]} features")

"""
======================================================
Save VIF-Filtered Data (for Linear Regression)
======================================================
"""
print(f"\n3. Saving VIF-filtered data (for Linear Regression)...")
X_train_vif.to_csv("/Users/skygastinel/indeng/secondhand-clothing-sales/data/X_train_clean_encoded.csv", index=False)
X_test_vif.to_csv("/Users/skygastinel/indeng/secondhand-clothing-sales/data/X_test_clean_encoded.csv", index=False)

print("\n" + "="*70)
print("DATA SAVING COMPLETE")
print("="*70)
print("\nSummary:")
print(f"  • FULL data ({X_train.shape[1]} features) → X_train/test_clean_encoded_FULL.csv")
print(f"  • VIF-filtered ({X_train_vif.shape[1]} features) → X_train/test_clean_encoded.csv")
print(f"  • {len(vif_col_drop)} features removed for Linear Regression due to multicollinearity")
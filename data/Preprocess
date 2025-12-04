import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0

df = pd.read_csv("vestiaire.csv")

# Drop rows with missing target
df = df.dropna(subset=["price_usd"])

# Map conditions to ordinal
condition_map = {
    "Never worn, with tag": 7,
    "Never worn": 6,          
    "Very good condition": 5,
    "Good condition": 4,
    "Satisfactory condition": 3,
    "Fair condition": 2,
    "Poor condition": 1,
}

df["condition_score"] = df["product_condition"].map(condition_map)

# Missing indicator
df["condition_missing"] = df["condition_score"].isna().astype(int)

# Fill missing with 0
df["condition_score"] = df["condition_score"].fillna(0)



def clean_color(x):
    # If it's missing (NaN), give it a special category
    if pd.isna(x):
        return "unknown"
    x = str(x).lower().strip()
    # If it’s multi-word like "Black Navy", take the first color
    x = x.split()[0]
    return x

df["color_clean"] = df["product_color"].apply(clean_color)


def parse_ship_time(x):
    #convert string "1-2" days into numbers
    if pd.isna(x):
        return np.nan
    s = str(x)
    # find all numbers in the string
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return np.nan
    nums = [float(n) for n in nums]
    # e.g. '1-2 days' -> (1+2)/2 = 1.5
    return sum(nums) / len(nums)

df["usually_ships_within_days"] = df["usually_ships_within"].apply(parse_ship_time)

# brand frequency-encoded, (Too many brands for one-hot, this is just the most simple solution I can think of, can be changed later)
df["brand_popularity"] = df["brand_name"].map(
    df["brand_name"].value_counts(normalize=True)
)

target = "price_usd"

numeric_features = [
    "product_like_count",        
    "usually_ships_within_days",
    "seller_products_sold",
    "seller_num_products_listed",
    "seller_num_followers",
    "seller_pass_rate",
    "condition_score",
    "condition_missing",
    "brand_popularity",
]

cats = [
    "product_gender_target",
    "product_category",
    "product_type",
    "product_season",
    "product_material",
    "color_clean",
    "seller_badge",
    "seller_country"
]

feature_cols = numeric_features + cats
# The features I did not include were ID categories, categories that were too wordy, and ones that were too related to price_usd

X = df[feature_cols].copy()
y = np.log1p(df["price_usd"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# remove outliers in price_usd for training set
train_prices = np.expm1(y_train) 
low = train_prices.quantile(0.001)
high = train_prices.quantile(0.995)
mask = (train_prices >= low) & (train_prices <= high)
X_train_clean = X_train.loc[mask].copy()
y_train_clean = y_train.loc[mask].copy()

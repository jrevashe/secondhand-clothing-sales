import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0

df = pd.read_csv("../vestiaire.csv")

# Drop rows with missing target
df = df.dropna(subset=["seller_price"])

# narrowed dataset down to women clothing only
df = df[(df["product_gender_target"] != "Men") & (df["product_category"].str.contains("Clothing", case=False, na=False))]

def clean_color(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).lower()

    # Standardize separators
    x = x.replace("/", " ").replace("&", " ")

    tokens = x.split()

    # If any token exactly matches a known color category, pick first match
    base_colors = [
        "black","blue","navy","brown","beige","green","pink","white",
        "red","grey","gray","yellow","orange","purple","gold","silver",
        "burgundy","metallic","multicolour","multicolor","turquoise"
    ]

    for t in tokens:
        if t in base_colors:
            return t

    # fallback: return first token
    return tokens[0]


df["color_clean"] = df["product_color"].apply(clean_color)


def parse_ship_time(x):
    #convert string 1-2 days into numbers
    if pd.isna(x):
        return np.nan
    s = str(x)
    # find all numbers in the string
    nums = re.findall(r"\d+\.?\d*", s)
    if not nums:
        return np.nan
    nums = [float(n) for n in nums]
    #1-2 days -> (1+2)/2 = 1.5
    return sum(nums) / len(nums)

df["usually_ships_within_days"] = df["usually_ships_within"].apply(parse_ship_time)

df.to_csv("../secondhand-clothing-sales/data/data_preprocessing/clean.csv")
df.info()

target = "seller_price"

numeric_features = [
    "product_like_count",
    "usually_ships_within_days",
    "seller_products_sold",
    "seller_num_products_listed",
    "seller_num_followers",
    "seller_pass_rate"
]

cats = [
    "product_type",
    "product_season",
    "product_material",
    "product_condition",
    "color_clean",
    "seller_badge",
    "seller_country",
    "brand_name",
]

feature_cols = numeric_features + cats

X = df[feature_cols].copy()
y = np.log1p(df["seller_price"])

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

#inspecting product_condition
print(f"X_Train lengthe: { len(X_train)}")
# #total matches number of rows = 38440
print(f"X_Train unique entries are: { X_train['product_condition'].unique()}")
print(f"X_Train unique entries count: { X_train['product_condition'].value_counts().sum()}")
print(f"X_Train missing entries count: { X_train['product_condition'].isna().sum()}")

print(f"X_test unique entries are: { X_test['product_condition'].unique()}")
print(f"X_test unique entries count: { X_test['product_condition'].value_counts().sum()}")
print(f"X_test missing entries count: { X_test['product_condition'].isna().sum()}")
#

###results:
# X_Train lengthe: 153757
# X_Train unique entries are: ['Never worn' 'Fair condition' 'Very good condition'
#  'Never worn, with tag' 'Good condition']
# X_Train unique entries count: 153757
# X_Train missing entries count: 0
# X_test length: 38440
# X_test unique entries are: ['Fair condition' 'Never worn' 'Very good condition'
#  'Never worn, with tag' 'Good condition']
# X_test unique entries count: 38440
# X_test missing entries count: 0





#### NOTE!!!!!!! Replace path with your correct local path
X_train_clean.to_csv("../secondhand-clothing-sales/data/data_preprocessing/X_train_clean.csv", index=False)
y_train_clean.to_csv("/Users/skygastinel/indeng/secondhand-clothing-sales/data/data_preprocessing/y_train_clean.csv", index=False, header=False)

X_test.to_csv("../secondhand-clothing-sales/data/data_preprocessing/X_test_clean.csv", index=False)
y_test.to_csv("../secondhand-clothing-sales/data/data_preprocessing/y_test_clean.csv", index=False, header=False)
# print(f"X_train_clean: {X_train_clean}")
# print(f"y_train_clean: {y_train_clean}")
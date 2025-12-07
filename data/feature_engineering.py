import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

"""
======================================================
Load Datasets
======================================================
"""
#### NOTE: replace path with your correct local path
#load X_train and X_test
X_train = pd.read_csv("/Users/arevashe/secondhand-clothing-sales/data/X_train_clean.csv")
X_test = pd.read_csv("/Users/arevashe/secondhand-clothing-sales/data/X_test_clean.csv")

y_train = pd.read_csv("/Users/arevashe/secondhand-clothing-sales/data/y_train_clean.csv",header=None).squeeze("columns")


"""
======================================================
Initial Feature Engineering
======================================================
"""
#brand processing: for converting brands into tiers
brand_df = X_train.copy()
y_train = pd.to_numeric(y_train, errors="coerce")
brand_df["y"] = y_train

# Brand-level mean + count of log-price (train ONLY)
brand_stats = brand_df.groupby("brand_name")["y"].agg(["mean", "count"])

global_mean = y_train.mean()
k = 10  # smoothing strength

# Smoothed target encoding
brand_stats["smooth_te"] = (
    brand_stats["count"] * brand_stats["mean"] + k * global_mean
) / (brand_stats["count"] + k)

# Bin smoothed into tiers (Budget to Ultra-Luxury)
brand_stats["tier"] = pd.qcut(
    brand_stats["smooth_te"],
    q=5,
    labels=["Budget", "Mid-Market", "Premium", "Luxury", "Ultra-Luxury"]
)

# Mapping: brand_name -> tier label
brand_to_tier = brand_stats["tier"].to_dict()

# Apply to train/test; unseen brands get "Unknown" tier
X_train["brand_tier"] = X_train["brand_name"].map(brand_to_tier).fillna("Unknown")
X_test["brand_tier"]  = X_test["brand_name"].map(brand_to_tier).fillna("Unknown")

#remove raw brand_name column
X_train = X_train.drop(columns=["brand_name"])
X_test  = X_test.drop(columns=["brand_name"])



# Handle NaN shipping times, impute with median and create missing indicator
X_train["ships_missing"] = X_train["usually_ships_within_days"].isna().astype(int)
X_test["ships_missing"]  = X_test["usually_ships_within_days"].isna().astype(int)

median_ship = X_train["usually_ships_within_days"].median()

X_train["usually_ships_within_days"] = (
    X_train["usually_ships_within_days"].fillna(median_ship)
)
X_test["usually_ships_within_days"] = (
    X_test["usually_ships_within_days"].fillna(median_ship)
)
"""
======================================================
Functions
======================================================
"""

#categorical columns to encode: product_type, product_season, product_material, color_clean, seller_badge, seller_country

#general logic: create a function for each column; bring each entry to lower case, group common features together one one-hot encode these. refer to comment for individual functions on the one hot encode key

def transform_product_type(entry):
    """
    This function transforms the product_type column
    Key for One Hot Encoding - {1: outerwear, 2: bottoms, 3: top, 4: swimwear, 5: dress, 6: one-piece}
    """
    if pd.isna(entry):
        return 0
    entry = entry.lower()
    #list specifc categories first so they are picked up first by the function
    if "jacket" in entry or "coat" in entry or "vest" in entry or "cardigan" in entry  or "blazer" in entry or "sweatshirt" in entry  or "jumper" in entry  or "knitwear" in entry or "puffer" in entry or "caban" in entry or "peacoat" in entry:
        
        return 1
    if "pant" in entry or "trouser" in entry or "legging" in entry or "jean" in entry or "short" in entry  or "tight" in entry or "bermuda" in entry or "skirt" in entry:
        
        return 2
    if "top" in entry or "shirt" in entry or "blouse" in entry or "corset" in entry:
        
        return 3
    if "swimsuit" in entry or "bikini" in entry:
        
        return 4
    if "dress" in entry:
        return 5
    if "jumpsuit" in entry or "romper" in entry :
        return 6
    else:
        return 0

def transform_product_season(entry):
    """
    This function transforms the product_season column
    Key for One Hot Encoding - {1: all seasons, 2: autumn/winter, 3: spring/summer}
    """
    if pd.isna(entry):
        return 0
    entry = entry.lower()
    if "all seasons" in entry:
        
        return 1
    if "autumn / winter" in entry:
        
        return 2
    if "spring / summer" in entry:
        
        return 3
    else:
        return 0
    
def transform_product_material(entry):
    """
    This function transforms the product_material column
    Key for One Hot Encoding - {1: viscose, 2: cotton, 3: synthetic, 4: polyester, 5: polyamide, 6: wool, 7: leather, 8: silk,
    9: denim, 10: suede, 11: khaki, 12: linen, 13: cashmere, 14: velvet, 15: lace, 16: fur, 17: glitter, 18: tweed, 19: vinyl,
    20: lycra, 21: spandex, 22: shearling, 23: rubber, 24: plastic, 25: cloth, 26: sponge, 27: animal}

    """
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
    #could be faux fur but generaized it - note this in limitations section
    if "fur" in entry:
        
        return  16
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
    if  'python' in entry or 'fox' in entry or 'rabbit' in entry or 'water snake' in entry or 'mink' in entry or 'beaver' in entry or 'mongolian lamb' in entry or 'raccoon' in entry or 'crocodile'in entry or 'astrakhan' in entry or 'ostrich' in entry or 'eel' in entry or 'pony' in entry or  'chinchilla' in entry or 'alligator' in entry:
        
        return 27
    else:
        return 0

def transform_color_clean(entry):
    """
    This function transforms the color_clean column
    Key for One Hot Encoding - {1: metallic colors, 2: multicolour, 3: blue, 4: black, 5: brown, 6: green, 7: pink, 8: white, 9: red, 10: grey,
     11: yellow, 12: orange, 13: purple, 14: turquoise}

    """
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
    
    else:
        return 0
def transform_seller_badge(entry):
    """
    This function transforms the seller_badg column
    Key for One Hot Encoding - {1: common, 2: trusted, 3: expert}

    """
    if pd.isna(entry):
        return 0
    entry = entry.lower()
    #{1: common, 2: trusted, 3: expert}
    if "common" in entry:
        
        return 1
    if "trusted" in entry:
        
        return 2
    if "expert" in entry:
        
        return 3
    
    else:
        return 0

def transform_seller_country(entry):
    """
    This function transforms the seller_country column
    Key for One Hot Encoding - {1: europe, 2: asia, 3: north america, 4:south america, 5: africa, 6: oceania, 7: carriabean}

    """
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
    else:
        return 0

def transform_product_condition(entry):
    """
    This function transforms the product_condition column to ordinal variables
    Key for One Hot Encoding - condition_map = {
                               "Never worn, with tag": 7,
                               "Never worn": 6,
                               "Very good condition": 5,
                               "Good condition": 4,
                               "Satisfactory condition": 3,
                               "Fair condition": 2,
                               "Poor condition": 1,
                               }

    """
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
    else:
        return 0



# df["condition_score"] = df["product_condition"].map(condition_map)
#
# # Missing indicator
# df["condition_missing"] = df["condition_score"].isna().astype(int)
#
# # Fill missing with 0
# df["condition_score"] = df["condition_score"].fillna(0)

"""
======================================================
Transforming Columns for X_train
======================================================
"""

# product_type
X_train.info()
X_train["cleaned_product_type"] = X_train["product_type"].apply(transform_product_type)
print(f"product_type encoded count is: {X_train['cleaned_product_type'].value_counts().sum()}")

# product_season
X_train["cleaned_product_season"] = X_train["product_season"].apply(transform_product_season)
print(f"product_season encoded count is: {X_train['cleaned_product_season'].value_counts().sum()}")

# product_material
X_train["cleaned_product_material"] = X_train["product_material"].apply(transform_product_material)
print(f"product_material encoded count is: {X_train['cleaned_product_material'].value_counts().sum()}")
# print(f"unique entries are: { X_train["product_material"].unique()}")

#color_clean
# print(f"unique color_clean entries are: { X_train["color_clean"].unique()}")
X_train["cleaned_color_clean"] = X_train["color_clean"].apply(transform_color_clean)
print(f"color_clean encoded count is: {X_train['cleaned_color_clean'].value_counts().sum()}")

# seller_badge
# print(f"unique seller_badge entries are: { X_train["seller_badge"].unique()}")
X_train["cleaned_seller_badge"] = X_train["seller_badge"].apply(transform_seller_badge)
print(f"seller_badge encoded count is: {X_train['cleaned_seller_badge'].value_counts().sum()}")

# seller_country
# print(f"unique seller_country entries are: { X_train["seller_country"].unique()}")
X_train["cleaned_seller_country"] = X_train["seller_country"].apply(transform_seller_country)
print(f"seller_country encoded count is: {X_train['cleaned_seller_country'].value_counts().sum()}")

X_train["cleaned_product_condition"] = X_train["product_condition"].apply(transform_product_condition)
(f"product_condition encoded count is: {X_train['product_condition'].value_counts().sum()}")

#drop categorical columns
X_train = X_train.drop(columns=["product_type", "product_season", "product_material", "color_clean", "seller_badge", "seller_country", "product_condition"])

#one-hot encode categorical columns
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

X_train = pd.get_dummies(X_train, columns=one_hot_cols, prefix=one_hot_cols,  drop_first=True)

#write df to csv
X_train.to_csv("/Users/arevashe/secondhand-clothing-sales/data/X_train_clean_encoded.csv", index=False)

"""
======================================================
Transforming Columns for X_test
======================================================
"""

# product_type
X_test.info()
X_test["cleaned_product_type"] = X_test["product_type"].apply(transform_product_type)
print(f"product_type encoded count is: {X_test['cleaned_product_type'].value_counts().sum()}")

# product_season
X_test["cleaned_product_season"] = X_test["product_season"].apply(transform_product_season)
print(f"product_season encoded count is: {X_test['cleaned_product_season'].value_counts().sum()}")

# product_material
X_test["cleaned_product_material"] = X_test["product_material"].apply(transform_product_material)
print(f"product_material encoded count is: {X_test['cleaned_product_material'].value_counts().sum()}")
# print(f"unique entries are: { X_train["product_material"].unique()}")

#color_clean
# print(f"unique color_clean entries are: { X_test["color_clean"].unique()}")
X_test["cleaned_color_clean"] = X_test["color_clean"].apply(transform_color_clean)
print(f"color_clean encoded count is: {X_test['cleaned_color_clean'].value_counts().sum()}")

# seller_badge
# print(f"unique seller_badge entries are: { X_test["seller_badge"].unique()}")
X_test["cleaned_seller_badge"] = X_test["seller_badge"].apply(transform_seller_badge)
print(f"seller_badge encoded count is: {X_test['cleaned_seller_badge'].value_counts().sum()}")

# seller_country
# print(f"unique seller_country entries are: { X_test["seller_country"].unique()}")
X_test["cleaned_seller_country"] = X_test["seller_country"].apply(transform_seller_country)
print(f"seller_country encoded count is: {X_test['cleaned_seller_country'].value_counts().sum()}")

X_test["cleaned_product_condition"] = X_test["product_condition"].apply(transform_product_condition)
print(f"product_condition encoded count is: {X_test['product_condition'].value_counts().sum()}")


#drop categorical columns
X_test = X_test.drop(columns=["product_type", "product_season", "product_material", "color_clean", "seller_badge", "seller_country", "product_condition"])

#one-hot encode categorical columns
X_test = pd.get_dummies(X_test, columns=one_hot_cols, prefix=one_hot_cols, drop_first=True).astype(int)

#align columns of X_test to X_train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


"""
======================================================
Dropping Columns based on most recent VIF test
======================================================
"""

###VERSION 2 results

#Analysis:

#removing cleaned_product_type_1, cleaned_product_season_2, cleaned_product_material_6, cleaned_product_condition_5, cleaned_product_condition_6, cleaned_color_clean_4 (>10)
#                         columns  VIF values
# 0            product_like_count    1.342643
# 1     usually_ships_within_days    3.099087
# 2          seller_products_sold    5.099565
# 3    seller_num_products_listed    4.409493
# 4          seller_num_followers    1.970550
# 5              seller_pass_rate    9.033228
# 6                 ships_missing    2.101386
# 7        cleaned_product_type_1   11.396265
# 8        cleaned_product_type_2    7.831167
# 9        cleaned_product_type_3    5.937864
# 10       cleaned_product_type_4    1.968336
# 11       cleaned_product_type_5    9.509251
# 12       cleaned_product_type_6    1.417397
# 13     cleaned_product_season_2   11.442688
# 14     cleaned_product_season_3    3.067936
# 15   cleaned_product_material_2    4.150194
# 16   cleaned_product_material_3    1.423131
# 17   cleaned_product_material_4    2.509228
# 18   cleaned_product_material_5    1.487115
# 19   cleaned_product_material_6   10.276211
# 20   cleaned_product_material_7    1.423098
# 21   cleaned_product_material_8    1.928738
# 22   cleaned_product_material_9    1.504274
# 23  cleaned_product_material_10    1.044285
# 24  cleaned_product_material_12    2.298893
# 25  cleaned_product_material_13    2.732003
# 26  cleaned_product_material_14    1.072011
# 27  cleaned_product_material_15    1.085240
# 28  cleaned_product_material_16    1.383733
# 29  cleaned_product_material_17    1.056245
# 30  cleaned_product_material_18    1.058022
# 31  cleaned_product_material_19    1.008882
# 32  cleaned_product_material_20    1.227778
# 33  cleaned_product_material_21    1.019194
# 34  cleaned_product_material_22    1.096739
# 35  cleaned_product_material_23    1.000747
# 36  cleaned_product_material_24    1.001282
# 37  cleaned_product_material_25    1.019886
# 38  cleaned_product_material_26    1.002180
# 39  cleaned_product_material_27    1.279382
# 40  cleaned_product_condition_4    5.470951
# 41  cleaned_product_condition_5   27.673903
# 42  cleaned_product_condition_6   20.445740
# 43        cleaned_color_clean_2    6.251990
# 44        cleaned_color_clean_3    8.381520
# 45        cleaned_color_clean_4   12.267607
# 46        cleaned_color_clean_5    8.069521
# 47        cleaned_color_clean_6    2.781698
# 48        cleaned_color_clean_7    3.644768
# 49        cleaned_color_clean_8    4.812992
# 50        cleaned_color_clean_9    3.252902
# 51       cleaned_color_clean_10    3.869243
# 52       cleaned_color_clean_11    1.776550
# 53       cleaned_color_clean_12    1.688949
# 54       cleaned_color_clean_13    1.866940
# 55       cleaned_color_clean_14    1.290664
# 56     cleaned_seller_country_2    1.165185
# 57     cleaned_seller_country_3    1.223186
# 58     cleaned_seller_country_4    1.000577
# 59     cleaned_seller_country_5    1.000894
# 60     cleaned_seller_country_6    1.026096
# 61     cleaned_seller_country_7    1.000481
# 62       cleaned_seller_badge_2    1.814590
# 63       cleaned_seller_badge_3    2.360442
# 64            brand_tier_Luxury    3.711203
# 65        brand_tier_Mid-Market    2.754503
# 66           brand_tier_Premium    5.468890
# 67      brand_tier_Ultra-Luxury    1.940272



vif_col_drop = ["cleaned_product_type_1", "cleaned_product_season_2", "cleaned_product_material_6", "cleaned_product_condition_5","cleaned_product_condition_6", "cleaned_color_clean_4"]

X_train = X_train.drop(columns=vif_col_drop)
X_test = X_test.drop(columns=vif_col_drop)

"""
======================================================
Write df to csv
======================================================
"""

#### NOTE: replace path with your correct local path
#write df to csv
X_train.to_csv("/Users/arevashe/secondhand-clothing-sales/data/X_train_clean_encoded.csv", index=False)
X_test.to_csv("/Users/arevashe/secondhand-clothing-sales/data/X_test_clean_encoded.csv", index=False)

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

"""
======================================================
Load Datasets
======================================================
"""

#load X_train and X_test
X_train = pd.read_csv("/Users/arevashe/secondhand-clothing-sales/data/X_train_clean.csv")
X_test = pd.read_csv("/Users/arevashe/secondhand-clothing-sales/data/X_test_clean.csv")

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
    count = 0
    entry = entry.lower()
    #list specifc categories first so they are picked up first by the function
    if "jacket" in entry or "coat" in entry or "vest" in entry or "cardigan" in entry  or "blazer" in entry or "sweatshirt" in entry  or "jumper" in entry  or "knitwear" in entry or "puffer" in entry or "caban" in entry or "peacoat" in entry:
        count += 1
        return 1
    if "pant" in entry or "trouser" in entry or "legging" in entry or "jean" in entry or "short" in entry  or "tight" in entry or "bermuda" in entry or "skirt" in entry:
        count += 1
        return 2
    if "top" in entry or "shirt" in entry or "blouse" in entry or "corset" in entry:
        count += 1
        return 3
    if "swimsuit" in entry or "bikini" in entry:
        count += 1
        return 4
    if "dress" in entry:
        count += 1
        return 5
    if "jumpsuit" in entry or "romper" in entry :
        count += 1
        return 6


def transform_product_season(entry):
    """
    This function transforms the product_season column
    Key for One Hot Encoding - {1: all seasons, 2: autumn/winter, 3: spring/summer}
    """
    count = 0
    entry = entry.lower()
    if "all seasons" in entry:
        count += 1
        return 1
    if "autumn / winter" in entry:
        count += 1
        return 2
    if "spring / summer" in entry:
        count += 1
        return 3

def transform_product_material(entry):
    """
    This function transforms the product_material column
    Key for One Hot Encoding - {1: viscose, 2: cotton, 3: synthetic, 4: polyester, 5: polyamide, 6: wool, 7: leather, 8: silk,
    9: denim, 10: suede, 11: khaki, 12: linen, 13: cashmere, 14: velvet, 15: lace, 16: fur, 17: glitter, 18: tweed, 19: vinyl,
    20: lycra, 21: spandex, 22: shearling, 23: rubber, 24: plastic, 25: cloth, 26: sponge, 27: animal}

    """
    count = 0
    entry = entry.lower()

    if "viscose" in entry:
        count += 1
        return 1
    if "cotton" in entry:
        count += 1
        return 2
    if "synthetic" in entry:
        count += 1
        return 3
    if "polyester" in entry:
        count += 1
        return 4
    if "polyamide" in entry:
        count += 1
        return 5
    if "wool" in entry:
        count += 1
        return 6
    if "leather" in entry:
        count += 1
        return 7
    if "silk" in entry:
        count += 1
        return 8
    if "denim" in entry:
        count += 1
        return 9
    if "suede" in entry:
        count += 1
        return 10
    if "khaki" in entry:
        count += 1
        return 11
    if "linen" in entry:
        count += 12
        return "linen"
    if "cashmere" in entry:
        count += 1
        return 13
    if "velvet" in entry:
        count += 1
        return 14
    if "lace" in entry:
        count += 1
        return 15
    #could be faux fur but generaized it - note this in limitations section
    if "fur" in entry:
        count += 1
        return  16
    if "glitter" in entry:
        count += 1
        return 17
    if "tweed" in entry:
        count += 1
        return 18
    if "vinyl" in entry:
        count += 1
        return 19
    if "lycra" in entry:
        count += 1
        return 20
    if "spandex" in entry:
        count += 1
        return 21
    if "shearling" in entry:
        count += 1
        return 22
    if "rubber" in entry:
        count += 1
        return 23
    if "plastic" in entry:
        count += 1
        return 24
    if "cloth" in entry:
        count += 1
        return 25
    if "sponge" in entry:
        count += 1
        return 26
    if  'python' in entry or 'fox' in entry or 'rabbit' in entry or 'water snake' in entry or 'mink' in entry or 'beaver' in entry or 'mongolian lamb' in entry or 'raccoon' in entry or 'crocodile'in entry or 'astrakhan' in entry or 'ostrich' in entry or 'eel' in entry or 'pony' in entry or  'chinchilla' in entry or 'alligator' in entry:
        count += 1
        return 27


def transform_color_clean(entry):
    """
    This function transforms the color_clean column
    Key for One Hot Encoding - {1: metallic colors, 2: multicolour, 3: blue, 4: black, 5: brown, 6: green, 7: pink, 8: white, 9: red, 10: grey,
     11: yellow, 12: orange, 13: purple, 14: turquoise}

    """
    count = 0
    entry = entry.lower()
    if "gold" in entry or "silver" in entry or "metallic" in entry:
        count += 1
        return 1
    if "multicolour" in entry:
        count += 1
        return 2
    if "blue" in entry or "navy" in entry:
        count += 1
        return 3
    if "black" in entry:
        count += 1
        return 4
    if "brown" in entry or "beige" in entry or "ecru" in entry or "khaki" in entry or "camel" in entry:
        count += 1
        return 5
    if "green" in entry:
        count += 1
        return 6
    if "pink" in entry:
        count += 1
        return 7
    if "white" in entry:
        count += 1
        return 8
    if "red" in entry or "burgundy" in entry:
        count += 1
        return 9
    if "grey" in entry or "anthracite" in entry:
        count += 1
        return 10
    if "yellow" in entry:
        count += 1
        return 11
    if "orange" in entry:
        count += 1
        return 12
    if "purple" in entry:
        count += 1
        return 13
    if "turquoise" in entry:
        count += 1
        return 14

def transform_seller_badge(entry):
    """
    This function transforms the seller_badg column
    Key for One Hot Encoding - {1: expert, 2: common, 3: trusted}

    """
    count = 0
    entry = entry.lower()
    #{1: expert, 2: common, 3: trusted}
    if "expert" in entry:
        count += 1
        return 1
    if "common" in entry:
        count += 1
        return 2
    if "trusted" in entry:
        count += 1
        return 3

def transform_seller_country(entry):
    """
    This function transforms the seller_country column
    Key for One Hot Encoding - {1: europe, 2: asia, 3: north america, 4:south america, 5: africa, 6: oceania, 7: carriabean}

    """
    count = 0
    if "United Kingdom" in entry or "Cyprus" in entry or "France" in entry or "Italy" in entry or "Switzerland" in entry or "Belgium" in entry or "Hungary" in entry or "Spain" in entry or "Netherlands" in entry or "Austria" in entry or "Latvia" in entry or "Sweden" in entry or "Finland" in entry or "Germany" in entry or "Poland" in entry or "Romania" in entry or "Denmark" in entry or "Lithuania" in entry or "Portugal" in entry or "Greece" in entry or "Luxembourg" in entry or "Ireland" in entry or "Monaco" in entry or "Bulgaria" in entry or "Czech Republic" in entry or "Slovenia" in entry or "Slovakia" in entry or "Norway" in entry or "Croatia" in entry or "Estonia" in entry or "Albania" in entry or "Malta" in entry or "Liechtenstein" in entry or "Ukraine" in entry or "Belarus" in entry or "Andorra" in entry or "Serbia" in entry or "Isle Of Man" in entry:
        count += 1
        return 1
    if "Indonesia" in entry or "Japan" in entry or "Saudi Arabia" in entry or "Hong Kong" in entry or "Turkey" in entry or "Lebanon" in entry or "Singapore" in entry or "South Korea" in entry or "China" in entry or "Philippines" in entry or "Israel" in entry or "Russia" in entry or "Malaysia" in entry or "Thailand" in entry or "Bahrain" in entry or "Mongolia" in entry or "Taiwan" in entry or "Kazakhstan" in entry or "Qatar" in entry or "Kuwait" in entry or "Macau" in entry or "United Arab Emirates" in entry:
        count += 1
        return 2
    if "United States" in entry or "Canada" in entry or "Mexico" in entry:
        count += 1
        return 3
    if "Brazil" in entry:
        count += 1
        return 4
    if "Nigeria" in entry or "Tunisia" in entry or "The Canary Islands" in entry:
        count += 1
        return 5
    if "Australia" in entry or "New Zealand" in entry:
        count += 1
        return 6
    if "St Barthelemy" in entry or "Guadeloupe" in entry:
        count += 1
        return 7

"""
======================================================
Transforming Columns for X_train
======================================================
"""

# product_type
X_train.info()
X_train["cleaned_product_type"] = X_train["product_type"].apply(transform_product_type)
X_train = X_train[X_train["cleaned_product_type"].notna()]
print(f"product_type encoded count is: {X_train["cleaned_product_type"].value_counts().sum()}")

# product_season
X_train["cleaned_product_season"] = X_train["product_season"].apply(transform_product_season)
print(f"product_season encoded count is: {X_train["cleaned_product_season"].value_counts().sum()}")

# product_material
X_train["cleaned_product_material"] = X_train["product_material"].apply(transform_product_material)
print(f"product_material encoded count is: {X_train["cleaned_product_material"].value_counts().sum()}")
# print(f"unique entries are: { X_train["product_material"].unique()}")

#color_clean
# print(f"unique color_clean entries are: { X_train["color_clean"].unique()}")
X_train["cleaned_color_clean"] = X_train["color_clean"].apply(transform_color_clean)
print(f"color_clean encoded count is: {X_train["cleaned_color_clean"].value_counts().sum()}")

# seller_badge
# print(f"unique seller_badge entries are: { X_train["seller_badge"].unique()}")
X_train["cleaned_seller_badge"] = X_train["seller_badge"].apply(transform_seller_badge)
print(f"seller_badge encoded count is: {X_train["cleaned_seller_badge"].value_counts().sum()}")

# seller_country
# print(f"unique seller_country entries are: { X_train["seller_country"].unique()}")
X_train["cleaned_seller_country"] = X_train["seller_country"].apply(transform_seller_country)
print(f"seller_country encoded count is: {X_train["cleaned_seller_country"].value_counts().sum()}")

#drop categorical columns
X_train = X_train.drop(columns=["product_type", "product_season", "product_material", "color_clean", "seller_badge", "seller_country"])

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
X_test = X_test[X_test["cleaned_product_type"].notna()]
print(f"product_type encoded count is: {X_test["cleaned_product_type"].value_counts().sum()}")

# product_season
X_test["cleaned_product_season"] = X_test["product_season"].apply(transform_product_season)
print(f"product_season encoded count is: {X_test["cleaned_product_season"].value_counts().sum()}")

# product_material
X_test["cleaned_product_material"] = X_test["product_material"].apply(transform_product_material)
print(f"product_material encoded count is: {X_test["cleaned_product_material"].value_counts().sum()}")
# print(f"unique entries are: { X_train["product_material"].unique()}")

#color_clean
# print(f"unique color_clean entries are: { X_test["color_clean"].unique()}")
X_test["cleaned_color_clean"] = X_test["color_clean"].apply(transform_color_clean)
print(f"color_clean encoded count is: {X_test["cleaned_color_clean"].value_counts().sum()}")

# seller_badge
# print(f"unique seller_badge entries are: { X_test["seller_badge"].unique()}")
X_test["cleaned_seller_badge"] = X_test["seller_badge"].apply(transform_seller_badge)
print(f"seller_badge encoded count is: {X_test["cleaned_seller_badge"].value_counts().sum()}")

# seller_country
# print(f"unique seller_country entries are: { X_test["seller_country"].unique()}")
X_test["cleaned_seller_country"] = X_test["seller_country"].apply(transform_seller_country)
print(f"seller_country encoded count is: {X_test["cleaned_seller_country"].value_counts().sum()}")

#drop categorical columns
X_test = X_test.drop(columns=["product_type", "product_season", "product_material", "color_clean", "seller_badge", "seller_country"])

#write df to csv
X_test.to_csv("/Users/arevashe/secondhand-clothing-sales/data/X_test_clean_encoded.csv", index=False)


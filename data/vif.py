import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
======================================================
Load Datasets
======================================================
"""
#### NOTE: replace path with your correct local path
#load X_train and X_test
X_train = pd.read_csv("/Users/arevashe/secondhand-clothing-sales/data/X_train_clean_encoded.csv")
X_test = pd.read_csv("/Users/arevashe/secondhand-clothing-sales/data/X_test_clean_encoded.csv")


"""
======================================================
VIF
======================================================
"""

model = sm.add_constant(X_train)

vif_dataframe = pd.DataFrame()
vif_dataframe["columns"]  = X_train.columns
vif_dataframe["VIF values"]  = [variance_inflation_factor(X_train, x) for x in range(X_train.shape[1])]

print(vif_dataframe.to_string())

###INTITAL results

# Analysis:
#product_condition should have been ohe after being transformed to ordinal variable, further inspection showed no product condiiton was missing or NaN
#brand popularity likely is perfectly multicolinear (liekly bc it was derived using other columns), so drop this
#droppin any other columns with a score > 10 (after NaN values have been taken care of)

#
#                         columns  VIF values
# 0            product_like_count    1.342210
# 1     usually_ships_within_days    3.092211
# 2          seller_products_sold    5.036617
# 3    seller_num_products_listed    4.353742
# 4          seller_num_followers    1.958183
# 5              seller_pass_rate    8.991015
# 6               condition_score   28.548038
# 7             condition_missing         NaN
# 8              brand_popularity         NaN
# 9                 ships_missing    2.091436
# 10       cleaned_product_type_1   10.662964
# 11       cleaned_product_type_2    7.429484
# 12       cleaned_product_type_3    5.608684
# 13       cleaned_product_type_4    1.970235
# 14       cleaned_product_type_5    8.954525
# 15       cleaned_product_type_6    1.393068
# 16     cleaned_product_season_2   11.430427
# 17     cleaned_product_season_3    3.037559
# 18   cleaned_product_material_2    4.095914
# 19   cleaned_product_material_3    1.413931
# 20   cleaned_product_material_4    2.492227
# 21   cleaned_product_material_5    1.481610
# 22   cleaned_product_material_6   10.271738
# 23   cleaned_product_material_7    1.416873
# 24   cleaned_product_material_8    1.908403
# 25   cleaned_product_material_9    1.498744
# 26  cleaned_product_material_10    1.043133
# 27  cleaned_product_material_12    2.296814
# 28  cleaned_product_material_13    2.730570
# 29  cleaned_product_material_14    1.070530
# 30  cleaned_product_material_15    1.083032
# 31  cleaned_product_material_16    1.383361
# 32  cleaned_product_material_17    1.054035
# 33  cleaned_product_material_18    1.057088
# 34  cleaned_product_material_19    1.008868
# 35  cleaned_product_material_20    1.226870
# 36  cleaned_product_material_21    1.018997
# 37  cleaned_product_material_22    1.096724
# 38  cleaned_product_material_23    1.000738
# 39  cleaned_product_material_24    1.001292
# 40  cleaned_product_material_25    1.019674
# 41  cleaned_product_material_26    1.002164
# 42  cleaned_product_material_27    1.278686
# 43        cleaned_color_clean_2    5.654218
# 44        cleaned_color_clean_3    7.543909
# 45        cleaned_color_clean_4   10.978901
# 46        cleaned_color_clean_5    7.275401
# 47        cleaned_color_clean_6    2.591257
# 48        cleaned_color_clean_7    3.366157
# 49        cleaned_color_clean_8    4.425986
# 50        cleaned_color_clean_9    2.991702
# 51       cleaned_color_clean_10    3.517259
# 52       cleaned_color_clean_11    1.696130
# 53       cleaned_color_clean_12    1.616902
# 54       cleaned_color_clean_13    1.769510
# 55       cleaned_color_clean_14    1.259897
# 56     cleaned_seller_country_2    1.158672
# 57     cleaned_seller_country_3    1.223312
# 58     cleaned_seller_country_4    1.000559
# 59     cleaned_seller_country_5    1.000894
# 60     cleaned_seller_country_6    1.026041
# 61     cleaned_seller_country_7    1.000469
# 62       cleaned_seller_badge_2    1.812889
# 63       cleaned_seller_badge_3    2.360981
# 64            brand_tier_Luxury    3.648831
# 65        brand_tier_Mid-Market    2.725286
# 66           brand_tier_Premium    5.387343
# 67      brand_tier_Ultra-Luxury    1.925007


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



###VERSION 3 Results

#Analysis:
#This was to check VIF values after columns had been dropped. All columns show a score <10, so this confirms dataset passes VIF test
#
#                         columns  VIF values
# 0            product_like_count    1.332987
# 1     usually_ships_within_days    2.793858
# 2          seller_products_sold    5.061798
# 3    seller_num_products_listed    4.366837
# 4          seller_num_followers    1.960685
# 5              seller_pass_rate    7.398990
# 6                 ships_missing    1.706108
# 7        cleaned_product_type_2    1.749366
# 8        cleaned_product_type_3    1.586930
# 9        cleaned_product_type_4    1.850787
# 10       cleaned_product_type_5    1.946079
# 11       cleaned_product_type_6    1.045988
# 12     cleaned_product_season_3    2.669212
# 13   cleaned_product_material_2    2.374187
# 14   cleaned_product_material_3    1.182418
# 15   cleaned_product_material_4    1.620530
# 16   cleaned_product_material_5    1.236054
# 17   cleaned_product_material_7    1.141812
# 18   cleaned_product_material_8    1.474496
# 19   cleaned_product_material_9    1.295468
# 20  cleaned_product_material_10    1.018152
# 21  cleaned_product_material_12    2.072098
# 22  cleaned_product_material_13    1.115953
# 23  cleaned_product_material_14    1.027899
# 24  cleaned_product_material_15    1.039642
# 25  cleaned_product_material_16    1.025212
# 26  cleaned_product_material_17    1.026914
# 27  cleaned_product_material_18    1.024315
# 28  cleaned_product_material_19    1.003762
# 29  cleaned_product_material_20    1.206595
# 30  cleaned_product_material_21    1.011566
# 31  cleaned_product_material_22    1.007556
# 32  cleaned_product_material_23    1.000585
# 33  cleaned_product_material_24    1.000424
# 34  cleaned_product_material_25    1.013634
# 35  cleaned_product_material_26    1.001317
# 36  cleaned_product_material_27    1.019417
# 37  cleaned_product_condition_4    1.128656
# 38        cleaned_color_clean_2    1.440719
# 39        cleaned_color_clean_3    1.667533
# 40        cleaned_color_clean_5    1.544214
# 41        cleaned_color_clean_6    1.135631
# 42        cleaned_color_clean_7    1.209213
# 43        cleaned_color_clean_8    1.353366
# 44        cleaned_color_clean_9    1.168903
# 45       cleaned_color_clean_10    1.213909
# 46       cleaned_color_clean_11    1.062666
# 47       cleaned_color_clean_12    1.054783
# 48       cleaned_color_clean_13    1.066944
# 49       cleaned_color_clean_14    1.024656
# 50     cleaned_seller_country_2    1.154724
# 51     cleaned_seller_country_3    1.213893
# 52     cleaned_seller_country_4    1.000569
# 53     cleaned_seller_country_5    1.000814
# 54     cleaned_seller_country_6    1.025107
# 55     cleaned_seller_country_7    1.000453
# 56       cleaned_seller_badge_2    1.787214
# 57       cleaned_seller_badge_3    2.309877
# 58            brand_tier_Luxury    2.820804
# 59        brand_tier_Mid-Market    2.202975
# 60           brand_tier_Premium    4.053009
# 61      brand_tier_Ultra-Luxury    1.651650
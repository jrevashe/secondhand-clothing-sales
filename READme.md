# UC Berkeley 142A Final Project (Secondhand Clothing Price Prediction)

**Requirements**


#Recommended (easy)

Run 'pip install -r requirements.txt'

#Manual Installation

Run 'pip install numpy pandas scikit-learn statsmodels matplotlib seaborn'

===============================

Git LFS was used for pushing the original dataset due to its large size. Ensure Git LFS is installed before cloning the repository.

brew install git-lfs
git lfs install


**Clone Git Repository**


Open terminal on your local machine
Navigate to the directory to store project

Run 'git clone https://github.com/jrevashe/secondhand-clothing-sales.git'

Navigate to repository directory on terminal and open the repository on preferred IDE


**Steps**


Data Preprocessing and Feature Engineering
1) Run data/data_preprocessing/data_preprocessing.py
2) Run data/feature_engineering/feature_engineering.py
3) Run data/vif.py - to verify results (not needed for training)

Training
1) Run models/linear_regression/linear_regression.py
   a) Results saved to following files and terminal
2) Run models/cart/cart.py
   b) Results saved to following files and terminal
3) Run models/gradient_boosting/gradient_boosting.py
   c) Results saved to following files and terminal


**Directory Structure**
```text
<tree>
secondhand-clothing-sales/
│
├── README.md
├── .gitattributes
├── requirements.txt
│
├── data/
│   ├── vestaire.csv
│   ├── vif.py
│   ├── data_preprocessing/
│   │   ├── clean.csv
│   │   ├── data_preprocess.py
│   │   ├── X_test_clean.csv
│   │   ├── X_train_clean.csv
│   │   ├── y_test_clean.csv
│   │   └── y_train_clean.csv
│   └── feature_engineering/
│       ├── feature_engineering.py
│       ├── X_test_clean_encoded.csv
│       ├── X_train_clean_encoded.csv
│       ├── X_test_clean_encoded_FULL.csv
│       └── X_train_clean_encoded_FULL.csv
│
└── models/
    ├── cart/
    │   ├── cart.py
    │   ├── cart_feature_importance.csv
    │   ├── cart_predictions.csv
    │   ├── cart_results.csv
    │   ├── cart_summary.txt
    │   └── cart_tree_visualization.png
    │
    ├── gradient_boosting/
    │   ├── gradient_boosting.py
    │   ├── gradient_boosting_feature_importance.csv
    │   ├── gradient_boosting_predictions.csv
    │   ├── gradient_boosting_results.csv
    │   └── gradient_boosting_summary.txt
    │
    └── linear_regression/
        ├── linear_regression.py
        ├── linear_regression_feature_importance.csv
        ├── linear_regression_predictions.csv
        ├── linear_regression_results.png
        └── linear_regression_summary.txt

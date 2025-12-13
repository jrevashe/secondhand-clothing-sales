
"""
Gradient Boosting Model for Secondhand Clothing Price Prediction

This model uses...
- Hyperparameter tuning via cross-validation
- Early stopping to prevent overfitting
- Learning curve analysis
- Feature importance analysis
- Model evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import time

# random seed for reproducibility
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

################################################################################
# 1. Load Data
################################################################################

print("=" * 70)
print("GRADIENT BOOSTING MODEL")
print("=" * 70)
print("\nLoading data...")

# Load preprocessed and encoded datasets
X_train = pd.read_csv("../data/X_train_clean_encoded_FULL.csv")
X_test = pd.read_csv("../data/X_test_clean_encoded_FULL.csv")
y_train = pd.read_csv("../data/y_train_clean.csv", header=None).squeeze("columns")
y_test = pd.read_csv("../data/y_test_clean.csv", header=None).squeeze("columns")

print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
print(f"Number of features: {X_train.shape[1]}")

################################################################################
# 2. Hyperparameter Tuning with Grid Search Cross-Validation
################################################################################

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING - GRID SEARCH WITH 5-FOLD CV")
print("=" * 70)

# Using a focused parameter grid for gradient boosting
param_grid = {
    'n_estimators': [200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [20],
    'min_samples_leaf': [10],
    'subsample': [0.8],
    'max_features': ['sqrt', None]
}

print("\nParameter Grid:")
print(f"n_estimators: {param_grid['n_estimators']}")
print(f"learning_rate: {param_grid['learning_rate']}")
print(f"max_depth: {param_grid['max_depth']}")
print(f"min_samples_split: {param_grid['min_samples_split']}")
print(f"min_samples_leaf: {param_grid['min_samples_leaf']}")
print(f"subsample: {param_grid['subsample']}")
print(f"max_features: {param_grid['max_features']}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nTotal parameter combinations to test: {total_combinations}")

# Initialize gradient boosting model
gb_base = GradientBoostingRegressor(random_state=RANDOM_STATE, verbose=0)

#Grid Search with 5-fold cross-validation
print("\nPerforming Grid Search (may take several minutes)...")

start_time = time.time()

grid_search = GridSearchCV(
    estimator=gb_base,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

elapsed_time = time.time() - start_time
print(f"\nGrid Search completed in {elapsed_time/60:.2f} minutes")

print("\n" + "=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

# Get best model
best_gb = grid_search.best_estimator_

################################################################################
# 3. Training with Early Stopping (using validation set)
################################################################################
print("\n" + "=" * 70)
print("TRAINING WITH EARLY STOPPING")
print("=" * 70)

# Validation set from training data (80-20 split)
from sklearn.model_selection import train_test_split
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
)

print(f"\nTraining set: {X_train_fit.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")

# Train model with early stopping (initially was taking a very long time to run)
print("\nTraining with early stopping (patience=20 rounds)...")

# Use best parameters from grid search
final_gb = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=grid_search.best_params_['learning_rate'],
    max_depth=grid_search.best_params_['max_depth'],
    min_samples_split=grid_search.best_params_['min_samples_split'],
    min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
    subsample=grid_search.best_params_['subsample'],
    max_features=grid_search.best_params_['max_features'],
    random_state=RANDOM_STATE,
    verbose=0,
    validation_fraction=0.2,
    n_iter_no_change=20,
    tol=0.0001
)

final_gb.fit(X_train, y_train)

print(f"\nOptimal number of estimators: {final_gb.n_estimators_}")
print(f"Training stopped at iteration: {final_gb.n_estimators_}")

################################################################################
# 4. Predictions
################################################################################
print("\nMaking predictions...")
y_train_pred = final_gb.predict(X_train)
y_test_pred = final_gb.predict(X_test)

# Transform predictions back to original scale
y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)
y_train_pred_original = np.expm1(y_train_pred)
y_test_pred_original = np.expm1(y_test_pred)


################################################################################
# 5. Model Evaluation
################################################################################
print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)

# Metrics on log scale
train_rmse_log = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse_log = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae_log = mean_absolute_error(y_train, y_train_pred)
test_mae_log = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nPerformance on Log-Transformed Target:")
print(f"  Training RMSE: {train_rmse_log:.4f}")
print(f"  Test RMSE:     {test_rmse_log:.4f}")
print(f"  Training MAE:  {train_mae_log:.4f}")
print(f"  Test MAE:      {test_mae_log:.4f}")
print(f"  Training R²:   {train_r2:.4f}")
print(f"  Test R²:       {test_r2:.4f}")

# Calculate metrics on original price scale
train_rmse_original = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
test_rmse_original = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
train_mae_original = mean_absolute_error(y_train_original, y_train_pred_original)
test_mae_original = mean_absolute_error(y_test_original, y_test_pred_original)

print("\nPerformance on Original Price Scale (USD):")
print(f"Training RMSE: ${train_rmse_original:.2f}")
print(f"Test RMSE:     ${test_rmse_original:.2f}")
print(f"Training MAE:  ${train_mae_original:.2f}")
print(f"Test MAE:      ${test_mae_original:.2f}")

# Calculate MAPE
train_mape = np.mean(np.abs((y_train_original - y_train_pred_original) / y_train_original)) * 100
test_mape = np.mean(np.abs((y_test_original - y_test_pred_original) / y_test_original)) * 100

print(f"Training MAPE: {train_mape:.2f}%")
print(f"Test MAPE:     {test_mape:.2f}%")

################################################################################
# 6. Feature Importance Analysis
################################################################################
print("\n" + "=" * 70)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 70)

# Create DataFrame of features + importance
feature_names = X_train.columns
importances = final_gb.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

importance_df_sorted = importance_df.sort_values('Importance', ascending=False)

# Top 15 features
print("\nTop 15 Features:")
for idx, row in importance_df_sorted.head(15).iterrows():
    print(f"  {row['Feature']:40s}: {row['Importance']:7.4f}")

################################################################################
# 7. Learning Curves (Training Progress)
################################################################################
print("\n" + "=" * 70)
print("LEARNING CURVE ANALYSIS")
print("=" * 70)

# Get training and validation scores at each boosting iteration
train_scores = []
test_scores = []


for i, (train_pred, test_pred) in enumerate(zip(
    final_gb.staged_predict(X_train),
    final_gb.staged_predict(X_test))):
        train_scores.append(mean_squared_error(y_train, train_pred))
        test_scores.append(mean_squared_error(y_test, test_pred))

best_iteration = np.argmin(test_scores)
print(f"\nBest iteration: {best_iteration + 1}")
print(f"Best test MSE: {test_scores[best_iteration]:.4f}")

################################################################################
# 8. Residual Analysis
################################################################################
print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)


train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

print(f"\nTraining residuals - Mean: {np.mean(train_residuals):.6f}, Std: {np.std(train_residuals):.4f}")
print(f"Test residuals - Mean: {np.mean(test_residuals):.6f}, Std: {np.std(test_residuals):.4f}")

################################################################################
# 9. Visualizations
################################################################################
print("\nGenerating visualizations...")

sns.set_style("whitegrid")

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 12))

# Actual vs Predicted on Log Scale
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_test_pred, alpha=0.5, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Log(Price)')
ax1.set_ylabel('Predicted Log(Price)')
ax1.set_title(f'Actual vs Predicted (Test Set, Log Scale)\nR² = {test_r2:.4f}')
ax1.grid(True, alpha=0.3)

# Actual vs Predicted on Original Scale
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test_original, y_test_pred_original, alpha=0.5, s=10)
ax2.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Price ($)')
ax2.set_ylabel('Predicted Price ($)')
ax2.set_title(f'Actual vs Predicted (Test Set, Original Scale)\nMAE = ${test_mae_original:.2f}')
ax2.grid(True, alpha=0.3)

# Residual Plot
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test_pred, test_residuals, alpha=0.5, s=10)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Log(Price)')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot (Test Set)')
ax3.grid(True, alpha=0.3)

# Top 15 Feature Importances
ax4 = plt.subplot(2, 3, 4)
top_15_features = importance_df_sorted.head(15)
ax4.barh(range(len(top_15_features)), top_15_features['Importance'], color='forestgreen')
ax4.set_yticks(range(len(top_15_features)))
ax4.set_yticklabels(top_15_features['Feature'], fontsize=9)
ax4.set_xlabel('Importance')
ax4.set_title('Top 15 Feature Importances')
ax4.grid(True, alpha=0.3, axis='x')

# Learning Curve
ax5 = plt.subplot(2, 3, 5)
iterations = range(1, len(train_scores) + 1)
ax5.plot(iterations, train_scores, label='Training MSE', linewidth=2)
ax5.plot(iterations, test_scores, label='Test MSE', linewidth=2)
ax5.axvline(x=best_iteration + 1, color='r', linestyle='--', label=f'Best iteration ({best_iteration + 1})')
ax5.set_xlabel('Boosting Iterations')
ax5.set_ylabel('Mean Squared Error')
ax5.set_title('Learning Curve (Training Progress)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Residual Distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(test_residuals, bins=50, alpha=0.7, color='forestgreen', edgecolor='black')
ax6.axvline(x=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Residuals')
ax6.set_ylabel('Frequency')
ax6.set_title('Residual Distribution (Test Set)')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('gradient_boosting_results.png', dpi=300, bbox_inches='tight')
print("Saved visualization as 'gradient_boosting_results.png'")

################################################################################
# 10. Save Results
################################################################################
print("\nSaving results...")

results_df = pd.DataFrame({
    'Actual_Log_Price': y_test,
    'Predicted_Log_Price': y_test_pred,
    'Actual_Price_USD': y_test_original,
    'Predicted_Price_USD': y_test_pred_original,
    'Residual': test_residuals
})
results_df.to_csv('gradient_boosting_predictions.csv', index=False)
print("Saved predictions to 'gradient_boosting_predictions.csv'")

# Feature importance
importance_df_sorted.to_csv('gradient_boosting_feature_importance.csv', index=False)
print("Saved feature importance to 'gradient_boosting_feature_importance.csv'")

# Model Summary
with open('gradient_boosting_summary.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("GRADIENT BOOSTING MODEL SUMMARY\n")
    f.write("IEOR 142A - Secondhand Clothing Price Prediction\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL CONFIGURATION:\n")
    f.write(f"Model Type: Gradient Boosting Regressor\n")
    f.write(f"Best Hyperparameters:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"    {param}: {value}\n")
    f.write(f"Number of Estimators: {final_gb.n_estimators_}\n")
    f.write(f"Number of Features: {X_train.shape[1]}\n")
    f.write(f"Training Samples: {X_train.shape[0]}\n")
    f.write(f"Test Samples: {X_test.shape[0]}\n\n")

    f.write("TRAINING DETAILS:\n")
    f.write(f"Best iteration: {best_iteration + 1}\n")
    f.write(f"Early stopping used: Yes (patience=20)\n")
    f.write(f"Grid search time: {elapsed_time/60:.2f} minutes\n\n")

    f.write("PERFORMANCE METRICS (Log Scale):\n")
    f.write(f"Training RMSE: {train_rmse_log:.4f}\n")
    f.write(f"Test RMSE:     {test_rmse_log:.4f}\n")
    f.write(f"Training MAE:  {train_mae_log:.4f}\n")
    f.write(f"Test MAE:      {test_mae_log:.4f}\n")
    f.write(f"Training R²:   {train_r2:.4f}\n")
    f.write(f"Test R²:       {test_r2:.4f}\n\n")

    f.write("PERFORMANCE METRICS (Original Scale - USD):\n")
    f.write(f"Training RMSE: ${train_rmse_original:.2f}\n")
    f.write(f"Test RMSE:     ${test_rmse_original:.2f}\n")
    f.write(f"Training MAE:  ${train_mae_original:.2f}\n")
    f.write(f"Test MAE:      ${test_mae_original:.2f}\n")
    f.write(f"Training MAPE: {train_mape:.2f}%\n")
    f.write(f"Test MAPE:     {test_mape:.2f}%\n\n")

    f.write("TOP 15 IMPORTANT FEATURES:\n")
    for idx, row in importance_df_sorted.head(15).iterrows():
        f.write(f"  {row['Feature']:40s}: {row['Importance']:7.4f}\n")

print("Model summary saved to 'gradient_boosting_summary.txt'")

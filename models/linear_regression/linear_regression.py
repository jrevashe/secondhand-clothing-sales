"""
Linear Regression Model for Secondhand Clothing Price Prediction
IEOR 142A Fall 2025 - Team Project

This script implements a linear regression model with:
- Ridge regularization (L2 penalty) to handle multicollinearity
- Cross-validation for hyperparameter tuning
- Feature standardization
- Model evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 70)
print("LINEAR REGRESSION MODEL - SECONDHAND CLOTHING PRICE PREDICTION")
print("=" * 70)
print("\nLoading data...")

# Load preprocessed and encoded datasets
X_train = pd.read_csv("../data/X_train_clean_encoded.csv")
X_test = pd.read_csv("../data/X_test_clean_encoded.csv")
y_train = pd.read_csv("../data/y_train_clean.csv", header=None).squeeze("columns")
y_test = pd.read_csv("../data/y_test_clean.csv", header=None).squeeze("columns")

print(f"Training set shape: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
print(f"Number of features: {X_train.shape[1]}")

# =============================================================================
# 2. Feature Scaling
# =============================================================================
print("\nStandardizing features...")
# Standardize features (important for Ridge regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 3. Model Training with Cross-Validation
# =============================================================================
print("\nTraining Ridge Regression with Cross-Validation...")
print("Testing alpha values: [0.001, 0.01, 0.1, 1, 10, 100, 1000]")

# Define alpha values to test
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Use RidgeCV for cross-validated Ridge regression
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

print(f"\nBest alpha (regularization parameter): {ridge_cv.alpha_}")

# Train final model with best alpha
best_model = Ridge(alpha=ridge_cv.alpha_, random_state=RANDOM_STATE)
best_model.fit(X_train_scaled, y_train)

# =============================================================================
# 4. Predictions
# =============================================================================
print("\nMaking predictions...")
y_train_pred = best_model.predict(X_train_scaled)
y_test_pred = best_model.predict(X_test_scaled)


# Transform predictions back to original scale
y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)
y_train_pred_original = np.expm1(y_train_pred)
y_test_pred_original = np.expm1(y_test_pred)

# =============================================================================
# 4. Stratified Results by Brand Tier
# =============================================================================

columns_tier = [column for column in X_test.columns if column.startswith("brand_tier_")]
columns_tier_updated = X_test[columns_tier].idxmax(axis=1).str.replace("brand_tier_", '')
for unique_tier in columns_tier_updated.unique():
    mask = columns_tier_updated == unique_tier

    y_true_unique_tier = y_test_original[mask]
    y_pred_unique_tier = y_test_pred_original[mask]

    r2_tier = r2_score(y_true_unique_tier, y_pred_unique_tier)
    mae_tier = mean_absolute_error(y_true_unique_tier, y_pred_unique_tier)
    mape_tier = mean_absolute_percentage_error(y_true_unique_tier, y_pred_unique_tier) * 100

    print("=" * 70)
    print(f"\n{unique_tier} Metrics Below ")
    print("=" * 70)
    print(f"sample size is {len(y_true_unique_tier)}")
    print(f"R squared is {r2_tier:.4f}")
    print(f"MAE is ${mae_tier:.2f}")
    print(f"MAPE is {mape_tier:.2f}%")
    print(f"average true price is ${y_true_unique_tier.mean():.2f}")

    print("=" * 70)
# =============================================================================
# 5. Model Evaluation
# =============================================================================
print("\n" + "=" * 70)
print("MODEL PERFORMANCE METRICS")
print("=" * 70)

# Calculate metrics on log scale
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

#moved above
# # Transform predictions back to original scale
# y_train_original = np.expm1(y_train)
# y_test_original = np.expm1(y_test)
# y_train_pred_original = np.expm1(y_train_pred)
# y_test_pred_original = np.expm1(y_test_pred)

# Calculate metrics on original price scale
train_rmse_original = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
test_rmse_original = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
train_mae_original = mean_absolute_error(y_train_original, y_train_pred_original)
test_mae_original = mean_absolute_error(y_test_original, y_test_pred_original)

print("\nPerformance on Original Price Scale (USD):")
print(f"  Training RMSE: ${train_rmse_original:.2f}")
print(f"  Test RMSE:     ${test_rmse_original:.2f}")
print(f"  Training MAE:  ${train_mae_original:.2f}")
print(f"  Test MAE:      ${test_mae_original:.2f}")

# Calculate MAPE (Mean Absolute Percentage Error)
train_mape = np.mean(np.abs((y_train_original - y_train_pred_original) / y_train_original)) * 100
test_mape = np.mean(np.abs((y_test_original - y_test_pred_original) / y_test_original)) * 100

print(f"  Training MAPE: {train_mape:.2f}%")
print(f"  Test MAPE:     {test_mape:.2f}%")

# =============================================================================
# 6. Feature Importance Analysis
# =============================================================================
print("\n" + "=" * 70)
print("TOP 15 MOST INFLUENTIAL FEATURES (by absolute coefficient)")
print("=" * 70)

# Get feature names and coefficients
feature_names = X_train.columns
coefficients = best_model.coef_

# Create DataFrame of features and their coefficients
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

# Sort by absolute value
coef_df_sorted = coef_df.sort_values('Abs_Coefficient', ascending=False)

# Display top 15 features
print("\nTop 15 Features:")
for idx, row in coef_df_sorted.head(15).iterrows():
    direction = "↑" if row['Coefficient'] > 0 else "↓"
    print(f"  {direction} {row['Feature']:40s}: {row['Coefficient']:7.4f}")

# =============================================================================
# 7. Residual Analysis
# =============================================================================
print("\n" + "=" * 70)
print("RESIDUAL ANALYSIS")
print("=" * 70)

# Calculate residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

print(f"\nTraining residuals - Mean: {np.mean(train_residuals):.6f}, Std: {np.std(train_residuals):.4f}")
print(f"Test residuals - Mean: {np.mean(test_residuals):.6f}, Std: {np.std(test_residuals):.4f}")

# =============================================================================
# 8. Visualizations
# =============================================================================
print("\nGenerating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted (Test Set) - Log Scale
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=10)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Log(Price)')
axes[0, 0].set_ylabel('Predicted Log(Price)')
axes[0, 0].set_title(f'Actual vs Predicted (Test Set, Log Scale)\nR² = {test_r2:.4f}')
axes[0, 0].grid(True, alpha=0.3)

# 2. Actual vs Predicted (Test Set) - Original Scale
axes[0, 1].scatter(y_test_original, y_test_pred_original, alpha=0.5, s=10)
axes[0, 1].plot([y_test_original.min(), y_test_original.max()],
                [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Price ($)')
axes[0, 1].set_ylabel('Predicted Price ($)')
axes[0, 1].set_title(f'Actual vs Predicted (Test Set, Original Scale)\nMAE = ${test_mae_original:.2f}')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residual Plot
axes[1, 0].scatter(y_test_pred, test_residuals, alpha=0.5, s=10)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Log(Price)')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residual Plot (Test Set)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Top 10 Feature Coefficients
top_10_features = coef_df_sorted.head(10)
colors = ['green' if c > 0 else 'red' for c in top_10_features['Coefficient']]
axes[1, 1].barh(range(len(top_10_features)), top_10_features['Coefficient'], color=colors)
axes[1, 1].set_yticks(range(len(top_10_features)))
axes[1, 1].set_yticklabels(top_10_features['Feature'], fontsize=9)
axes[1, 1].set_xlabel('Coefficient Value')
axes[1, 1].set_title('Top 10 Most Influential Features')
axes[1, 1].axvline(x=0, color='black', linestyle='-', lw=0.5)
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
print("Saved visualization as 'linear_regression_results.png'")

# =============================================================================
# 9. Save Results
# =============================================================================
print("\nSaving results...")

# Save predictions
results_df = pd.DataFrame({
    'Actual_Log_Price': y_test,
    'Predicted_Log_Price': y_test_pred,
    'Actual_Price_USD': y_test_original,
    'Predicted_Price_USD': y_test_pred_original,
    'Residual': test_residuals
})
results_df.to_csv('linear_regression_predictions.csv', index=False)
print("Saved predictions to 'linear_regression_predictions.csv'")

# Save feature importance
coef_df_sorted.to_csv('linear_regression_feature_importance.csv', index=False)
print("Saved feature importance to 'linear_regression_feature_importance.csv'")

# Save model summary
with open('linear_regression_summary.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("LINEAR REGRESSION MODEL SUMMARY\n")
    f.write("IEOR 142A - Secondhand Clothing Price Prediction\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL CONFIGURATION:\n")
    f.write(f"  Model Type: Ridge Regression (L2 Regularization)\n")
    f.write(f"  Best Alpha: {ridge_cv.alpha_}\n")
    f.write(f"  Number of Features: {X_train.shape[1]}\n")
    f.write(f"  Training Samples: {X_train.shape[0]}\n")
    f.write(f"  Test Samples: {X_test.shape[0]}\n\n")

    f.write("PERFORMANCE METRICS (Log Scale):\n")
    f.write(f"  Training RMSE: {train_rmse_log:.4f}\n")
    f.write(f"  Test RMSE:     {test_rmse_log:.4f}\n")
    f.write(f"  Training MAE:  {train_mae_log:.4f}\n")
    f.write(f"  Test MAE:      {test_mae_log:.4f}\n")
    f.write(f"  Training R²:   {train_r2:.4f}\n")
    f.write(f"  Test R²:       {test_r2:.4f}\n\n")

    f.write("PERFORMANCE METRICS (Original Scale - USD):\n")
    f.write(f"  Training RMSE: ${train_rmse_original:.2f}\n")
    f.write(f"  Test RMSE:     ${test_rmse_original:.2f}\n")
    f.write(f"  Training MAE:  ${train_mae_original:.2f}\n")
    f.write(f"  Test MAE:      ${test_mae_original:.2f}\n")
    f.write(f"  Training MAPE: {train_mape:.2f}%\n")
    f.write(f"  Test MAPE:     {test_mape:.2f}%\n\n")

    f.write("TOP 15 INFLUENTIAL FEATURES:\n")
    for idx, row in coef_df_sorted.head(15).iterrows():
        direction = "↑" if row['Coefficient'] > 0 else "↓"
        f.write(f"  {direction} {row['Feature']:40s}: {row['Coefficient']:7.4f}\n")

print("Saved model summary to 'linear_regression_summary.txt'")

print("\n" + "=" * 70)
print("LINEAR REGRESSION MODEL COMPLETE")
print("=" * 70)
print("\nKey Takeaways:")
print(f"  • The model achieves R² = {test_r2:.4f} on the test set")
print(f"  • Average prediction error: ${test_mae_original:.2f} (MAE)")
print(f"  • Mean percentage error: {test_mape:.2f}% (MAPE)")
print(f"  • Regularization parameter (alpha): {ridge_cv.alpha_}")
print("\nNext steps: Compare with CART and Gradient Boosting models")
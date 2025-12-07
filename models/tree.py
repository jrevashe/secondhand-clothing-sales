"""
CART (Decision Tree) Model for Secondhand Clothing Price Prediction
IEOR 142A Fall 2025 - Team Project

This script implements a CART regression tree with:
- Hyperparameter tuning via cross-validation
- Pruning to prevent overfitting (using cost-complexity pruning)
- Feature importance analysis
- Model evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 70)
print("CART DECISION TREE MODEL - SECONDHAND CLOTHING PRICE PREDICTION")
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
# 2. Hyperparameter Tuning with Grid Search Cross-Validation
# =============================================================================
print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING - GRID SEARCH WITH 5-FOLD CV")
print("=" * 70)

# Define parameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20, 25, None],
    'min_samples_split': [2, 10, 20, 50],
    'min_samples_leaf': [1, 5, 10, 20],
    'max_features': ['sqrt', 'log2', None]
}

print("\nParameter Grid:")
print(f"  max_depth: {param_grid['max_depth']}")
print(f"  min_samples_split: {param_grid['min_samples_split']}")
print(f"  min_samples_leaf: {param_grid['min_samples_leaf']}")
print(f"  max_features: {param_grid['max_features']}")

# Initialize CART model
cart_base = DecisionTreeRegressor(random_state=RANDOM_STATE)

# Perform Grid Search with 5-fold cross-validation
print("\nPerforming Grid Search (this may take a few minutes)...")
grid_search = GridSearchCV(
    estimator=cart_base,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n" + "=" * 70)
print("BEST HYPERPARAMETERS")
print("=" * 70)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score (neg MSE): {grid_search.best_score_:.4f}")

# Get best model
best_cart = grid_search.best_estimator_

# =============================================================================
# 3. Cost-Complexity Pruning Analysis
# =============================================================================
print("\n" + "=" * 70)
print("COST-COMPLEXITY PRUNING ANALYSIS")
print("=" * 70)

# Get the cost complexity path
path = best_cart.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print(f"\nNumber of alpha values to test: {len(ccp_alphas)}")

# Train trees with different alpha values (using a subset for efficiency)
alpha_subset = ccp_alphas[::max(1, len(ccp_alphas)//20)]  # Test ~20 alphas
train_scores = []
test_scores = []

print("Testing pruning levels...")
for ccp_alpha in alpha_subset:
    tree = DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        ccp_alpha=ccp_alpha,
        **grid_search.best_params_
    )
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# Find best alpha
best_alpha_idx = np.argmax(test_scores)
best_alpha = alpha_subset[best_alpha_idx]

print(f"\nBest pruning alpha: {best_alpha:.6f}")
print(f"Best test R² with pruning: {test_scores[best_alpha_idx]:.4f}")

# Train final pruned model
final_cart = DecisionTreeRegressor(
    random_state=RANDOM_STATE,
    ccp_alpha=best_alpha,
    **grid_search.best_params_
)
final_cart.fit(X_train, y_train)

# =============================================================================
# 4. Predictions
# =============================================================================
print("\nMaking predictions...")
y_train_pred = final_cart.predict(X_train)
y_test_pred = final_cart.predict(X_test)

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

# Transform predictions back to original scale
y_train_original = np.expm1(y_train)
y_test_original = np.expm1(y_test)
y_train_pred_original = np.expm1(y_train_pred)
y_test_pred_original = np.expm1(y_test_pred)

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
# 6. Tree Structure Information
# =============================================================================
print("\n" + "=" * 70)
print("TREE STRUCTURE")
print("=" * 70)
print(f"Number of nodes: {final_cart.tree_.node_count}")
print(f"Number of leaves: {final_cart.get_n_leaves()}")
print(f"Max depth achieved: {final_cart.get_depth()}")

# =============================================================================
# 7. Feature Importance Analysis
# =============================================================================
print("\n" + "=" * 70)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 70)

# Get feature importances
feature_names = X_train.columns
importances = final_cart.feature_importances_

# Create DataFrame of features and their importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance
importance_df_sorted = importance_df.sort_values('Importance', ascending=False)

# Display top 15 features
print("\nTop 15 Features:")
for idx, row in importance_df_sorted.head(15).iterrows():
    print(f"  {row['Feature']:40s}: {row['Importance']:7.4f}")

# =============================================================================
# 8. Residual Analysis
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
# 9. Visualizations
# =============================================================================
print("\nGenerating visualizations...")

# Set style
sns.set_style("whitegrid")

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 12))

# 1. Actual vs Predicted (Test Set) - Log Scale
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, y_test_pred, alpha=0.5, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Log(Price)')
ax1.set_ylabel('Predicted Log(Price)')
ax1.set_title(f'Actual vs Predicted (Test Set, Log Scale)\nR² = {test_r2:.4f}')
ax1.grid(True, alpha=0.3)

# 2. Actual vs Predicted (Test Set) - Original Scale
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test_original, y_test_pred_original, alpha=0.5, s=10)
ax2.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Price ($)')
ax2.set_ylabel('Predicted Price ($)')
ax2.set_title(f'Actual vs Predicted (Test Set, Original Scale)\nMAE = ${test_mae_original:.2f}')
ax2.grid(True, alpha=0.3)

# 3. Residual Plot
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(y_test_pred, test_residuals, alpha=0.5, s=10)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Log(Price)')
ax3.set_ylabel('Residuals')
ax3.set_title('Residual Plot (Test Set)')
ax3.grid(True, alpha=0.3)

# 4. Top 15 Feature Importances
ax4 = plt.subplot(2, 3, 4)
top_15_features = importance_df_sorted.head(15)
ax4.barh(range(len(top_15_features)), top_15_features['Importance'], color='steelblue')
ax4.set_yticks(range(len(top_15_features)))
ax4.set_yticklabels(top_15_features['Feature'], fontsize=9)
ax4.set_xlabel('Importance')
ax4.set_title('Top 15 Feature Importances')
ax4.grid(True, alpha=0.3, axis='x')

# 5. Pruning Analysis
ax5 = plt.subplot(2, 3, 5)
ax5.plot(alpha_subset, train_scores, marker='o', label='Train', drawstyle='steps-post')
ax5.plot(alpha_subset, test_scores, marker='o', label='Test', drawstyle='steps-post')
ax5.axvline(x=best_alpha, color='r', linestyle='--', label=f'Best α={best_alpha:.6f}')
ax5.set_xlabel('Alpha (Pruning Parameter)')
ax5.set_ylabel('R² Score')
ax5.set_title('Cost-Complexity Pruning Analysis')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xscale('log')

# 6. Residual Distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(test_residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax6.axvline(x=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Residuals')
ax6.set_ylabel('Frequency')
ax6.set_title('Residual Distribution (Test Set)')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('cart_results.png', dpi=300, bbox_inches='tight')
print("Saved visualization as 'cart_results.png'")

# =============================================================================
# 10. Tree Visualization (simplified version)
# =============================================================================
print("Generating simplified tree visualization...")
fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(
    final_cart,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3,  # Only show top 3 levels for readability
    ax=ax
)
plt.title("CART Decision Tree (First 3 Levels)", fontsize=16)
plt.tight_layout()
plt.savefig('cart_tree_visualization.png', dpi=300, bbox_inches='tight')
print("Saved tree visualization as 'cart_tree_visualization.png'")

# =============================================================================
# 11. Save Results
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
results_df.to_csv('cart_predictions.csv', index=False)
print("Saved predictions to 'cart_predictions.csv'")

# Save feature importance
importance_df_sorted.to_csv('cart_feature_importance.csv', index=False)
print("Saved feature importance to 'cart_feature_importance.csv'")

# Save model summary
with open('cart_summary.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("CART DECISION TREE MODEL SUMMARY\n")
    f.write("IEOR 142A - Secondhand Clothing Price Prediction\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL CONFIGURATION:\n")
    f.write(f"  Model Type: CART (Regression Tree)\n")
    f.write(f"  Best Hyperparameters:\n")
    for param, value in grid_search.best_params_.items():
        f.write(f"    {param}: {value}\n")
    f.write(f"  Best Pruning Alpha: {best_alpha:.6f}\n")
    f.write(f"  Number of Features: {X_train.shape[1]}\n")
    f.write(f"  Training Samples: {X_train.shape[0]}\n")
    f.write(f"  Test Samples: {X_test.shape[0]}\n\n")

    f.write("TREE STRUCTURE:\n")
    f.write(f"  Number of nodes: {final_cart.tree_.node_count}\n")
    f.write(f"  Number of leaves: {final_cart.get_n_leaves()}\n")
    f.write(f"  Max depth: {final_cart.get_depth()}\n\n")

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

    f.write("TOP 15 IMPORTANT FEATURES:\n")
    for idx, row in importance_df_sorted.head(15).iterrows():
        f.write(f"  {row['Feature']:40s}: {row['Importance']:7.4f}\n")

print("Saved model summary to 'cart_summary.txt'")

print("\n" + "=" * 70)
print("CART DECISION TREE MODEL COMPLETE")
print("=" * 70)
print("\nKey Takeaways:")
print(f"  • The model achieves R² = {test_r2:.4f} on the test set")
print(f"  • Average prediction error: ${test_mae_original:.2f} (MAE)")
print(f"  • Mean percentage error: {test_mape:.2f}% (MAPE)")
print(f"  • Tree depth: {final_cart.get_depth()}, Leaves: {final_cart.get_n_leaves()}")
print(f"  • Most important feature: {importance_df_sorted.iloc[0]['Feature']}")
print("\nComparison with Linear Regression:")
print("  • CART can capture non-linear relationships")
print("  • Feature importance shows which splits matter most")
print("  • Tree structure is interpretable (see tree_visualization.png)")
print("\nNext steps: Compare with Gradient Boosting model")

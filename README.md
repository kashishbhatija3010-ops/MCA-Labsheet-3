# MCA-Labsheet-3
Lab Assessment on Supervised Learning (Regression Models)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# 1
data = fetch_california_housing(as_frame=True)
df = data.frame   # Pandas DataFrame containing features + target ('MedHouseVal')

# 2
print("First 5 records:\n", df.head())
print("\nLast 5 records:\n", df.tail())

# 3
print("\nDataset Info:")
df.info()
print("\nDescriptive Statistics:\n", df.describe())

# 4
# Independent variables (X): all columns except target
# Dependent variable (y): MedHouseVal (Median House Value)
X_all = df.drop(columns=['MedHouseVal'])
y_all = df['MedHouseVal']
print("\nIndependent variables:", list(X_all.columns))
print("Dependent variable: MedHouseVal")

# 5
from sklearn.model_selection import train_test_split
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)
print(f"\nTraining set size: {X_train_all.shape[0]}, Testing set size: {X_test_all.shape[0]}")

# 6
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Using a single feature 'MedInc' (Median Income) to predict 'MedHouseVal'
X_simple = df[['MedInc']]
y_simple = df['MedHouseVal']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

# 6 
simple_lr = LinearRegression()

# 7
simple_lr.fit(X_train_s, y_train_s)

# 8
y_pred_simple = simple_lr.predict(X_test_s)

# 9
plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', alpha=0.4, label='Actual data')
plt.plot(X_test_s, y_pred_simple, color='red', linewidth=2, label='Regression line')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Simple Linear Regression: Income vs House Value')
plt.legend()
plt.savefig('simple_linear_regression.png')
plt.show()

# 10
comparison_df = pd.DataFrame({'Actual': y_test_s.values[:10], 'Predicted': y_pred_simple[:10]})
print("\nActual vs Predicted (Simple LR):\n", comparison_df)

# 11
print(f"\nCoefficient (slope): {simple_lr.coef_[0]:.4f}")
print(f"Intercept: {simple_lr.intercept_:.4f}")

# 12
new_income = np.array([[8.5]])   # e.g., median income = 8.5 (in tens of thousands)
predicted_value = simple_lr.predict(new_income)
print(f"\nPredicted house value for income=8.5: {predicted_value[0]:.4f}")


# 13
multi_lr = LinearRegression()

# 14
multi_lr.fit(X_train_all, y_train)

# 15
y_pred_multi = multi_lr.predict(X_test_all)

# 16
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_multi, alpha=0.4, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual House Value')
plt.ylabel('Predicted House Value')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.savefig('multiple_linear_regression.png')
plt.show()

# 17
coeff_df = pd.DataFrame({
    'Feature': X_all.columns,
    'Coefficient': multi_lr.coef_
}).sort_values(by='Coefficient', ascending=False)
print("\nEffect of each feature on prediction:\n", coeff_df)

# 18
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 18
poly2_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly2_model.fit(X_train_s, y_train_s)
print("Task 18: Polynomial Regression (Degree 2) model created and trained.")

# 19
poly3_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly3_model.fit(X_train_s, y_train_s)
print("Task 19: Polynomial Regression (Degree 3) model created and trained.")

# 20
y_pred_linear_compare = simple_lr.predict(X_test_s)
y_pred_poly2_compare = poly2_model.predict(X_test_s)
y_pred_poly3_compare = poly3_model.predict(X_test_s)

r2_linear = r2_score(y_test_s, y_pred_linear_compare)
r2_poly2 = r2_score(y_test_s, y_pred_poly2_compare)
r2_poly3 = r2_score(y_test_s, y_pred_poly3_compare)

print("Task 20: Comparison of R² scores")
print(f"Linear Regression R²      : {r2_linear:.4f}")
print(f"Polynomial Degree 2 R²    : {r2_poly2:.4f}")
print(f"Polynomial Degree 3 R²    : {r2_poly3:.4f}")

# 21
X_range = np.linspace(X_simple.min(), X_simple.max(), 300).reshape(-1, 1)
X_range_df = pd.DataFrame(X_range, columns=['MedInc'])

plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='gray', alpha=0.3, label='Actual data')
plt.plot(X_range_df, simple_lr.predict(X_range_df), color='blue', label='Linear')
plt.plot(X_range_df, poly2_model.predict(X_range_df), color='orange', label='Poly Degree 2')
plt.plot(X_range_df, poly3_model.predict(X_range_df), color='red', label='Poly Degree 3')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Linear vs Polynomial Regression Curves')
plt.legend()
plt.savefig('polynomial_regression.png')
plt.show()
print("Task 21: Polynomial regression curves visualized.")

# 22
y_pred_poly2 = poly2_model.predict(X_test_s)
y_pred_poly3 = poly3_model.predict(X_test_s)

print("Task 22: Predictions using Polynomial Regression models")
print("\nPoly Degree 2 - first 10 predictions:\n", y_pred_poly2[:10])
print("\nPoly Degree 3 - first 10 predictions:\n", y_pred_poly3[:10])

# 23
mae_poly2 = mean_absolute_error(y_test_s, y_pred_poly2)
mae_poly3 = mean_absolute_error(y_test_s, y_pred_poly3)
rmse_poly2 = np.sqrt(mean_squared_error(y_test_s, y_pred_poly2))
rmse_poly3 = np.sqrt(mean_squared_error(y_test_s, y_pred_poly3))

accuracy_comparison = pd.DataFrame({
    'Degree': [2, 3],
    'MAE': [mae_poly2, mae_poly3],
    'RMSE': [rmse_poly2, rmse_poly3],
    'R2': [r2_poly2, r2_poly3]
})

print("Task 23: Prediction accuracy comparison across polynomial degrees")
print(accuracy_comparison)

# 24-30

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print MAE, MSE, RMSE, R2 for a given model's predictions."""
    mae = mean_absolute_error(y_true, y_pred)          # Task 24
    mse = mean_squared_error(y_true, y_pred)            # Task 25
    rmse = np.sqrt(mse)                                  # Task 26
    r2 = r2_score(y_true, y_pred)                        # Task 27
    print(f"\n--- {model_name} ---")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    return {'Model': model_name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Task 28
results = []
results.append(evaluate_model(y_test_s, y_pred_simple, "Simple Linear Regression"))
results.append(evaluate_model(y_test, y_pred_multi, "Multiple Linear Regression"))
results.append(evaluate_model(y_test_s, y_pred_poly2, "Polynomial Regression (Degree 2)"))
results.append(evaluate_model(y_test_s, y_pred_poly3, "Polynomial Regression (Degree 3)"))

results_df = pd.DataFrame(results)
print("\nComparison Table:\n", results_df)

# 29
print("""
Interpretation:
- Lower MSE/RMSE indicates the model's predictions are closer to actual values (less error).
- R² closer to 1 means the model explains most of the variance in the target variable.
- R² closer to 0 (or negative) means the model performs poorly / no better than predicting the mean.
""")

# 30
errors = y_test - y_pred_multi
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_multi, errors, alpha=0.4, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Prediction Error (Actual - Predicted)')
plt.title('Prediction Errors Scatter Plot')
plt.savefig('prediction_errors.png')
plt.show()

# 31-35
from sklearn.preprocessing import StandardScaler
import joblib

# 31
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test_all)

scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)
y_pred_scaled = scaled_model.predict(X_test_scaled)

# 32
before_scaling = evaluate_model(y_test, y_pred_multi, "Before Scaling (Multiple LR)")
after_scaling = evaluate_model(y_test, y_pred_scaled, "After Scaling (Multiple LR)")

# 33
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True)
df_diabetes = diabetes.frame
X_diab = df_diabetes.drop(columns=['target'])
y_diab = df_diabetes['target']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42
)
diabetes_model = LinearRegression()
diabetes_model.fit(X_train_d, y_train_d)
y_pred_diab = diabetes_model.predict(X_test_d)
evaluate_model(y_test_d, y_pred_diab, "Diabetes Dataset - Linear Regression")

# 34
joblib.dump(multi_lr, 'multiple_linear_regression_model.joblib')
joblib.dump(scaler, 'feature_scaler.joblib')   # save scaler too, useful for future predictions
print("\nModel saved as 'multiple_linear_regression_model.joblib'")

# 35
loaded_model = joblib.load('multiple_linear_regression_model.joblib')
sample_input = X_test_all.iloc[[0]]   # take one sample row from test set
loaded_prediction = loaded_model.predict(sample_input)
print(f"\nPrediction using loaded model: {loaded_prediction[0]:.4f}")
print(f"Actual value for this sample: {y_test.iloc[0]:.4f}")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

file_path = "D:/KJWj/EXP1_Bostondata.csv"
df = pd.read_csv(file_path)
X = df.drop(columns="medv")
y = df["medv"]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 贝叶斯岭回归
br_model = BayesianRidge()
br_model.fit(X_train, y_train)

# 预测与评估
y_pred = br_model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="orange", label="Predictions", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
plt.xlabel("True House Prices")
plt.ylabel("Predicted House Prices")
plt.title(f"Bayesian Ridge Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

for name, coef in zip(X.columns, br_model.coef_):
    print(f"{name}: {coef:.4f}")
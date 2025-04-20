import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


file_path = "D:/KJWj/EXP1_Bostondata.csv"
df = pd.read_csv(file_path)
# 特征
X = df.drop(columns='medv')
y = df['medv']
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
# statsmodels OLS
X_with_const = sm.add_constant(X_scaled_df)
# 构建 OLS
model = sm.OLS(y, X_with_const).fit()
#预测与残差
y_pred = model.predict(X_with_const)
residuals = y - y_pred
print("R²:", round(model.rsquared, 4))
print("F-statistic:", round(model.fvalue, 2))
print("F p-value:", model.f_pvalue)

# 拟合线图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y, y=y_pred, marker='x', color='orange', label='预测点')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='理想拟合线')  # 对角线
plt.xlabel('True Prices (MEDV)')
plt.ylabel('Predicted Prices')
plt.title('Multivariate Linear Regression: True vs Predicted Prices')
plt.legend()
plt.tight_layout()
plt.show()
# 系数柱状图
coefficients = model.params.drop('const')
plt.figure(figsize=(10, 6))
coefficients.sort_values().plot(kind='barh', color='steelblue')
plt.title('Standardized Coefficients (Multivariate Linear Regression)')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.show()
# 残差图
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals, marker='x', color='black')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted MEDV')
plt.ylabel('Residuals')
plt.title('Residual Plot (Multivariate Linear Regression)')
plt.tight_layout()
plt.show()

# 回归摘要
print(model.summary())

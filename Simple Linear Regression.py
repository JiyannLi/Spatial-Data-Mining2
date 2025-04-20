import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_palette(["#2E4057", "#66A5AD", "#C4DFE6"])
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


file_path = "D:/KJWj/EXP1_Bostondata.csv"
df = pd.read_csv(file_path)
# 定义特征变量
features = ['crim', 'zn', 'dis', 'rad', 'black']
target = 'medv'
results = []


plt.figure(figsize=(18, 12))

# 遍历
for idx, feature in enumerate(features):
    X = df[[feature]]
    y = df[target]
    # 线性回归模型
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    # 评估
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    results.append((feature, r2, mse))


    plt.subplot(2, 3, idx + 1)
    sns.scatterplot(x=feature, y=target, data=df, label='Data', marker='x')
    plt.plot(X, y_pred, color='red', label='Fit')
    plt.title(f'{feature} vs {target}\nR² = {r2:.3f}, MSE = {mse:.2f}')
    plt.legend()

plt.tight_layout()
plt.suptitle("Simple Linear Regression Models", fontsize=16, y=1.02)
plt.show()

# 输出 R² 和 MSE
result_df = pd.DataFrame(results, columns=['Feature', 'R²', 'MSE'])
print(result_df)

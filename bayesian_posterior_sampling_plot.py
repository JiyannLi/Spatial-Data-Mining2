import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# （模拟真实线性关系 y = 3x + 噪声）
np.random.seed(42)
X_sim = np.linspace(0, 1, 100)[:, np.newaxis]
y_sim = 3 * X_sim.squeeze() + np.random.normal(0, 0.2, size=X_sim.shape[0])

# 抽样训练点
X_train_sim = X_sim[::5]
y_train_sim = y_sim[::5]
bayes_model = BayesianRidge()
bayes_model.fit(X_train_sim, y_train_sim)

# （不确定性）
y_mean, y_std = bayes_model.predict(X_sim, return_std=True)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(X_sim, y_sim, 'k--', label="True Function")
plt.scatter(X_train_sim, y_train_sim, c='red', label="Training Data")


for i in range(10):
    sample_weights = np.random.multivariate_normal(bayes_model.coef_, bayes_model.sigma_)
    y_sample = X_sim @ sample_weights + bayes_model.intercept_
    plt.plot(X_sim, y_sample, color='gray', alpha=0.3)

#
plt.plot(X_sim, y_mean, 'b', lw=2, label="Posterior Mean")
# 不确定性置信带（蓝色阴影）
#plt.fill_between(X_sim.squeeze(), y_mean - y_std, y_mean + y_std,
                 #color='blue', alpha=0.2, label="Uncertainty")

# 图像美化
plt.title("Bayesian Linear Regression: Posterior Sampling and Mean Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

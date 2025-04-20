
import pandas as pd
import pymc3 as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


file_path = "D:/KJWj/EXP1_Bostondata.csv"
df = pd.read_csv(file_path)


features = ["crim", "zn", "rm"]
X = df[features]
y = df["medv"].values
X_scaled = StandardScaler().fit_transform(X)

with pm.Model() as bayes_model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    betas = pm.Normal("betas", mu=0, sigma=10, shape=X_scaled.shape[1])
    sigma = pm.HalfNormal("sigma", sigma=10)

    mu = alpha + pm.math.dot(X_scaled, betas)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    trace = pm.sample(1000, return_inferencedata=True)

# 可视化
az.plot_trace(trace)
plt.tight_layout()
plt.show()

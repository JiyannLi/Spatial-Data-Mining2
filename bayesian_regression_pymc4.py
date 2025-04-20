import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def run_bayesian_regression():

    file_path = "D:/KJWj/EXP1_Bostondata.csv"
    df = pd.read_csv(file_path)
    features = ["crim", "zn", "rm", "dis", "rad", "tax", "lstat"]
    X = df[features].values
    y = df["medv"].values


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        betas = pm.Normal("betas", mu=0, sigma=5, shape=X_train.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=1)

        mu = intercept + pm.math.dot(X_train, betas)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_train_scaled)

        # 单核，避免多进程导致 spawn 错误
        trace = pm.sample(500, tune=500, target_accept=0.95, cores=1, return_inferencedata=True)

    return trace


if __name__ == "__main__":
    trace = run_bayesian_regression()

    # 后验轨迹图
    az.plot_trace(trace, figsize=(12, 8))
    plt.tight_layout()
    plt.show()

    # 森林图
    az.plot_forest(trace, var_names=["betas"], combined=True)
    plt.title("Posterior Distributions of Coefficients")
    plt.grid(True)
    plt.show()



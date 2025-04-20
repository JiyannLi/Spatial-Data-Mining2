import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


file_path = "D:/KJWj/EXP1_Bostondata.csv"
df = pd.read_csv(file_path)

# 二分类目标变量
df["target_class"] = (df["medv"] > 25).astype(int)
# 特征选择
features = ["crim", "zn", "rm", "dis", "rad", "tax", "lstat"]
X = df[features]
y = df["target_class"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 训练
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
# 预测评估
y_pred = nb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
labels = ["Low Price", "High Price"]

# 可视化
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Naive Bayes Classification")
plt.tight_layout()
plt.show()

report = classification_report(y_test, y_pred, output_dict=True)
metrics_df = pd.DataFrame(report).T.loc[["0", "1"], ["precision", "recall", "f1-score"]]
metrics_df.index = ["Low Price", "High Price"]

metrics_df.plot(kind='bar', figsize=(8, 5), colormap='viridis')
plt.title("Classification Metrics by Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()